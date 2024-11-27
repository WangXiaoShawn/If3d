import os
import sys

from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets

from networks.DDAM import DDAMNet
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--aff_path', type=str, default='Dataset/affectnet_smirk', help='AffectNet_smirk dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    return parser.parse_args() 
class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt+1
                    loss = loss+mse
            loss = cnt/(loss + eps)
        else:
            loss = 0
        return loss
class WarmUpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, base_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.base_lr = base_lr
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.base_lr - self.initial_lr) / float(self.warmup_epochs)
            return [self.initial_lr + warmup_factor * self.last_epoch for _ in self.optimizer.param_groups]
        return [self.base_lr for _ in self.optimizer.param_groups]
                
def run_training():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head)
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
    
    train_dataset = datasets.ImageFolder(f'{args.aff_path}/train', transform = data_transforms)   
    if args.num_class == 7:   # ignore the 8-th class
        idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] != 7]
        train_dataset = data.Subset(train_dataset, idx)

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle = False, 
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])      

    val_dataset = datasets.ImageFolder(f'{args.aff_path}/val', transform = data_transforms_val)  
    if args.num_class == 7:   # ignore the 8-th class 
        idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] != 7]
        val_dataset = data.Subset(val_dataset, idx)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)


    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    criterion_at = AttentionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=len(train_loader) // 2, mode='triangular',cycle_momentum=False)
    
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
                        
            out,feat,heads = model(imgs)

            loss = criterion_cls(out,targets)  + 0.1*criterion_at(heads)

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            scheduler.step()

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets in val_loader:
        
                imgs = imgs.to(device)
                targets = targets.to(device)
                out,feat,heads = model(imgs)

                loss = criterion_cls(out,targets)  + 0.1*criterion_at(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
            running_loss = running_loss/iter_cnt   
          

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))

            if args.num_class == 7 and  acc > 0.665:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('WhiteSMirkcheckpoints', "affecnet7_epoch"+str(epoch)+"_acc"+str(acc)+".pth"))
                tqdm.write('Model saved.')

            elif args.num_class == 8 and  acc > 0.4: 
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('PureSmirkcheckpoints', "PureSmirk_epoch"+str(epoch)+"_acc"+str(acc)+".pth"))
                tqdm.write('Model saved.')
        
if __name__ == "__main__":                    
    run_training()