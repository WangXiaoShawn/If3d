import os
import sys
import json
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler

from networks.DDAM_smirkpic_fusion import DDAMNet_Smirk_Spatial_Alignment_AttentionFusion
import pdb

import logging  # 新增
import sys      # 新增

# 定义 Logger 类
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--root1', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_clean'), help='Root directory for the first dataset.')
    parser.add_argument('--root2', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_smirk'), help='Root directory for the second dataset.')
    parser.add_argument('--train_excel_path', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_annotations/train_set_annotation_without_lnd.csv'), help='Path to Excel file containing training labels, val, and aro.')
    parser.add_argument('--val_excel_path', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_annotations/val_set_annotation_without_lnd.csv'), help='Path to Excel file containing validation labels, val, and aro.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--workers', default=24, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.') # 40->10
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')

    return parser.parse_args()

# ...（其他类定义保持不变）

def run_training():
    # 设置日志文件路径
    log_file = "training_log.txt"
    
    # 初始化日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 重定向 stdout 和 stderr
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)
    
    # 现在所有的 print 和 tqdm.write 都会被记录到日志文件中
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DDAMNet_Smirk_Spatial_Alignment_AttentionFusion(num_class=args.num_class, num_head=args.num_head)
    model.to(device)
        
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])
    
    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_sampler = ImbalancedDatasetSampler(CustomImageFolder(f'{args.root1}/train',  f'{args.root2}/train', args.train_excel_path, data_transforms))
    train_loader = get_dataloader(
        args, 
        f'{args.root1}/train', 
        f'{args.root2}/train',
        args.train_excel_path,
        data_transforms, 
        args.batch_size, 
        args.workers, 
        sampler=train_sampler
    )

    val_loader = get_dataloader(
        args, 
        f'{args.root1}/val', 
        f'{args.root2}/val', 
        args.val_excel_path,
        data_transforms_val, 
        args.batch_size, 
        args.workers, 
        shuffle=False
    )
    
    criterion_at = AttentionLoss()
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=len(train_loader) // 2, mode='triangular',cycle_momentum=False)
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for imgs, targets, aro_true, val_true in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            image1, image2 = imgs  # 解包
            image1, image2 = image1.float().to(device), image2.float().to(device)
            expression_true= targets.to(device).long() # cross entropy loss 要求long类型
            aro_true = aro_true.float().to(device)
            val_true = val_true.float().to(device)
            pred_expreesion, orig_heads_out, smirk_heads_out, pred_aro, pred_val, l3_loss = model(image1, image2)
            alpha, beta, gamma = np.random.rand(3)
            dloss = design_loss(val_true, aro_true, expression_true, pred_val, pred_aro, pred_expreesion, alpha, beta, gamma).float()
            criterion_orig = criterion_at(orig_heads_out).float()
            criterion_smirk = criterion_at(smirk_heads_out).float()
            loss =  dloss + 0.1 * criterion_orig + 0.1 * criterion_smirk + 0.1*l3_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicts = torch.max(pred_expreesion, 1)
            correct_num = torch.eq(predicts, expression_true).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(len(train_loader.dataset))
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
        logging.info('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr']))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets, aro_true, val_true  in val_loader:
                image1, image2 = imgs  
                image1, image2 = image1.float().to(device), image2.float().to(device)
                expression_true= targets.to(device).long() 
                aro_true = aro_true.float().to(device)
                val_true = val_true.float().to(device)
                pred_expression, orig_heads_out, smirk_heads_out, pred_aro, pred_val, l3_loss = model(image1, image2)
                dloss = design_loss(val_true, aro_true, expression_true, pred_val, pred_aro, pred_expression, alpha, beta, gamma).float()
                alpha, beta, gamma = np.random.rand(3)
                criterion_orig = criterion_at(orig_heads_out).float()
                criterion_smirk = criterion_at(smirk_heads_out).float()
                loss =  dloss + 0.1 * criterion_orig + 0.1 * criterion_smirk + 0.1*l3_loss
                running_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(pred_expression, 1)
                correct_num = torch.eq(predicts, expression_true).sum()
                bingo_cnt += correct_num.cpu()
                sample_cnt += pred_expression.size(0)

            running_loss = running_loss / iter_cnt
            scheduler.step() 
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            tqdm.write("[Epoch %d] Validation accuracy: %.4f. Loss: %.3f" % (epoch, acc, running_loss))
            logging.info("[Epoch %d] Validation accuracy: %.4f. Loss: %.3f" % (epoch, acc, running_loss))
            tqdm.write("best_acc: " + str(best_acc))
            logging.info("best_acc: " + str(best_acc))

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'iter': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join('picture_attention_fusion_va_attention_alignment_checkpoints', "affecnet_l3loss_epoch_" + str(epoch) + "_acc" + str(acc) + ".pth"))
                tqdm.write('Model saved.')
                logging.info('Model saved.')
