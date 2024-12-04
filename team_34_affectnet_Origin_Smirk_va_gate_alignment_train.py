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

import os
from networks.DDAM_smirkpic_fusion import DDAMNet_Smirk_Spatial_Alignment_GateFusion
import pdb


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

class CustomImageFolder(Dataset):
    def __init__(self, root1, root2, excel_path, transform=None, num_classes=8):
        self.root1 = root1
        self.root2 = root2
        self.transform = transform
        self.num_classes = num_classes
        self.labels_df = pd.read_csv(excel_path)  # 读取Excel文件
        self.imgs = self._load_images(root1, root2)
        print(f"Dataset initialized with {len(self.imgs)} samples")

    def _load_images(self, root1, root2):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        imgs = []
        paths1 = {}
        for subdir, _, files in os.walk(root1):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_extensions):
                    paths1[file] = os.path.join(subdir, file)
        
        for subdir, _, files in os.walk(root2):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_extensions):
                    if file in paths1:
                        path1 = paths1[file]
                        path2 = os.path.join(subdir, file)
                        label = self._get_label(subdir)
                        if label != -1:  # Ensure only valid labels are added
                            imgs.append((path1, path2, label))
                        else:
                            print(f"Invalid label for file {file} in {subdir}")
                    else:
                        print(f"File {file} in {subdir} not found in root1")
        
        print(f"Loaded {len(imgs)} image pairs")
        return imgs

    def _get_label(self, subdir):
        try:
            label = int(os.path.basename(subdir))
            if label < 0 or label >= self.num_classes:
                raise ValueError(f"Label {label} out of range for num_classes {self.num_classes}")
        except ValueError as e:
            print(f"Error parsing label from directory {subdir}: {e}")
            label = -1
        return label

    def __getitem__(self, index):
        if index >= len(self.imgs):
            print(f"Index {index} out of range for dataset of length {len(self.imgs)}")
            raise IndexError(f"Index {index} out of range for dataset of length {len(self.imgs)}")
        path1, path2, label = self.imgs[index]
        image1 = Image.open(path1).convert('RGB')
        image2 = Image.open(path2).convert('RGB')
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
    
        filename = os.path.basename(path1)
        number = int(os.path.splitext(filename)[0])
        row = self.labels_df[self.labels_df['number'] == number]
        if not row.empty:
            val = row['val'].values[0]
            aro = row['aro'].values[0]
        else:
            val = 0.0  # 或者其他默认值
            aro = 0.0  # 或者其他默认值
        
        return (image1, image2), label, val, aro

    def __len__(self):
        return len(self.imgs)

class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        # 采样数量
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()
        
        # 计算每个标签的权重
        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        # 获取数据集中的标签
        if isinstance(dataset, CustomImageFolder):
            labels = [x[2] for x in dataset.imgs]  # 确认标签在元组中的位置
        elif isinstance(dataset, torch.utils.data.Subset):
            labels = [dataset.dataset.imgs[i][2] for i in dataset.indices]
        else:
            raise NotImplementedError("Unsupported dataset type")
        return labels

    def __iter__(self):
        # 按权重随机抽样
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples




    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
    
class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt += 1
                    loss += mse
            loss = loss / cnt if cnt > 0 else 0
        else:
            loss = 0
        return loss
    
def design_loss(valence_true, arousal_true, expression_true, valence_pred, arousal_pred, expression_pred, alpha, beta, gamma):
    ce_loss = F.cross_entropy(expression_pred, expression_true)
    mse_valence = F.mse_loss(valence_pred, valence_true)
    mse_arousal = F.mse_loss(arousal_pred, arousal_true)
    mse = mse_valence + mse_arousal
    pcc_valence = torch.corrcoef(torch.stack([valence_true, valence_pred]))[0, 1] # 0行1列
    pcc_arousal = torch.corrcoef(torch.stack([arousal_true, arousal_pred]))[0, 1]
    pcc = (pcc_valence + pcc_arousal) / 2
    var_valence_true = torch.var(valence_true)
    var_valence_pred = torch.var(valence_pred)
    mean_valence_true = torch.mean(valence_true)
    mean_valence_pred = torch.mean(valence_pred)
    ccc_valence = 2.0 * var_valence_true * var_valence_pred * pcc_valence / (
        var_valence_true + var_valence_pred + (mean_valence_true - mean_valence_pred) ** 2.0)

    var_arousal_true = torch.var(arousal_true)
    var_arousal_pred = torch.var(arousal_pred)
    mean_arousal_true = torch.mean(arousal_true)
    mean_arousal_pred = torch.mean(arousal_pred)
    ccc_arousal = 2 * var_arousal_true * var_arousal_pred * pcc_arousal / (
        var_arousal_true + var_arousal_pred + (mean_arousal_true - mean_arousal_pred) ** 2)

    ccc = (ccc_valence + ccc_arousal) / 2
    return ce_loss + (alpha/(alpha+beta+gamma)) * mse +  (beta/(alpha+beta+gamma)) * (1 - pcc) +  (gamma/(alpha+beta+gamma))  * (1 - ccc)


def get_dataloader(args, root1, root2,excel_path ,transform, batch_size, workers, sampler=None, shuffle=False):
    dataset = CustomImageFolder(root1, root2,excel_path ,transform=transform)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,   
                                       num_workers=workers,
                                       sampler=sampler,
                                       pin_memory=True)

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DDAMNet_Smirk_Spatial_Alignment_GateFusion(num_class=args.num_class, num_head=args.num_head)
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
        for imgs, targets,aro_true, val_true in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            image1, image2 = imgs  # 解包
            image1, image2 = image1.float().to(device), image2.float().to(device)
            expression_true= targets.to(device).long() # cross entropy loss 要求long类型
            aro_true = aro_true.float().to(device)
            val_true = val_true.float().to(device)
            pred_expreesion, orig_heads_out, smirk_heads_out,pred_aro,pred_val,l3_loss= model(image1, image2)
            alpha, beta, gamma = np.random.rand(3)
            dloss = design_loss(val_true,aro_true,expression_true,pred_val,pred_aro,pred_expreesion,alpha,beta,gamma).float()
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

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets,aro_true, val_true  in val_loader:
                image1, image2 = imgs  
                image1, image2 = image1.float().to(device), image2.float().to(device)
                expression_true= targets.to(device).long() 
                aro_true = aro_true.float().to(device)
                val_true = val_true.float().to(device)
                pred_expression, orig_heads_out, smirk_heads_out,pred_aro,pred_val,l3_loss= model(image1, image2)
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
            tqdm.write("best_acc: " + str(best_acc))

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'iter': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join('picture_attention_fusion_va_gate_alignment_checkpoints', "affecnet_l3loss_epoch_" + str(epoch) + "_acc" + str(acc) + ".pth"))
                tqdm.write('Model saved.')

if __name__ == "__main__":
    run_training()
