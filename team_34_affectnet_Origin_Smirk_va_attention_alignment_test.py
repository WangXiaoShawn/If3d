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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from datetime import datetime

# 导入您的模型
from networks.DDAM_smirkpic_fusion import DDAMNet_Smirk_Spatial_Alignment_AttentionFusion

def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--root1', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_clean'), help='Root directory for the first dataset.')
    parser.add_argument('--root2', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_smirk'), help='Root directory for the second dataset.')
    parser.add_argument('--test_excel_path', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_annotations/val_set_annotation_without_lnd.csv'), help='Path to Excel file containing test labels, val, and aro.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--workers', default=24, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of classes.')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(base_dir, 'FinalCheckPoints/AttentionFusion.pth'), help='Path to the model checkpoint.')

    return parser.parse_args()

class CustomImageFolder(Dataset):
    def __init__(self, root1, root2, excel_path, transform=None, num_classes=8):
        self.root1 = root1
        self.root2 = root2
        self.transform = transform
        self.num_classes = num_classes
        self.labels_df = pd.read_csv(excel_path)
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
                        if label != -1:
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
            val = 0.0
            aro = 0.0

        return (image1, image2), label, val, aro

    def __len__(self):
        return len(self.imgs)

def get_dataloader(args, root1, root2, excel_path, transform, batch_size, workers, shuffle=False):
    dataset = CustomImageFolder(root1, root2, excel_path, transform=transform)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       num_workers=workers,
                                       shuffle=shuffle,
                                       pin_memory=True)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)  

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(format='%.2f')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

def evaluate():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet_Smirk_Spatial_Alignment_AttentionFusion(num_class=args.num_class, num_head=args.num_head)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_loader = get_dataloader(
        args,
        os.path.join(args.root1, 'val'),
        os.path.join(args.root2, 'val'),
        args.test_excel_path,
        data_transforms,
        args.batch_size,
        args.workers,
        shuffle=False
    )

    correct_sum = 0
    sample_cnt = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    all_targets = []
    all_predicted = []

    with torch.no_grad():
        for imgs, targets, aro_true, val_true in tqdm(test_loader, desc="Testing"):
            image1, image2 = imgs
            image1, image2 = image1.float().to(device), image2.float().to(device)
            expression_true = targets.to(device).long()
            aro_true = aro_true.float().to(device)
            val_true = val_true.float().to(device)

            pred_expression, _, _, pred_aro, pred_val, _ = model(image1, image2)
            loss = criterion(pred_expression, expression_true)
            total_loss += loss.item()

            _, predicts = torch.max(pred_expression, 1)
            correct_num = torch.eq(predicts, expression_true).sum()
            correct_sum += correct_num.cpu().numpy()
            sample_cnt += expression_true.size(0)

            # 收集所有的真实标签和预测标签
            all_targets.extend(expression_true.cpu().numpy())
            all_predicted.extend(predicts.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_sum / sample_cnt
    print(f"Test Accuracy: {accuracy * 100:.2f}%, Average Loss: {avg_loss:.4f}")

    # 生成混淆矩阵
    class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']
    matrix = confusion_matrix(all_targets, all_predicted)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=class8_names, normalize=True,
                          title='Attention_Fusion Confusion Matrix (acc: %0.2f%%)' % (accuracy * 100))

    # 保存混淆矩阵图像
    os.makedirs('FusionConfusionMatrix', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join('FusionConfusionMatrix', f"Attention_Fusion.png"))
    plt.close()

    # 计算其他分类指标
    precision = precision_score(all_targets, all_predicted, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predicted, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predicted, average='weighted', zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)  # 防止除以零

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(format='%.2f')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

if __name__ == "__main__":
    evaluate()
