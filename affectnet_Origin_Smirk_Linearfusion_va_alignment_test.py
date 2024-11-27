import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from networks.DDAM_smirkpic_fusion import DDAMNet_Smirk_Spatial_Alignment
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from datetime import datetime
import seaborn as sns
import argparse


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--root1', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_clean'), help='Root directory for the first dataset.')
    parser.add_argument('--root2', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_smirk'), help='Root directory for the second dataset.')
    parser.add_argument('--val_excel_path', type=str, default=os.path.join(base_dir, 'Dataset/affectnet_annotations/val_set_annotation_without_lnd.csv'), help='Path to Excel file containing validation labels, val, and aro.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(base_dir, 'FinalCheckPoints/LinearFusion.pth'), help='Path to the model checkpoint.')
    return parser.parse_args()

# 自定义数据集与训练代码类似
class CustomImageFolder(Dataset):
    def __init__(self, root1, root2, excel_path, transform=None, num_classes=8):
        self.root1 = root1
        self.root2 = root2
        self.transform = transform
        self.num_classes = num_classes
        self.labels_df = pd.read_csv(excel_path)  # 读取Excel文件
        self.imgs = self._load_images(root1, root2)

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
        return imgs

    def _get_label(self, subdir):
        try:
            label = int(os.path.basename(subdir))
            if label < 0 or label >= self.num_classes:
                raise ValueError(f"Label {label} out of range for num_classes {self.num_classes}")
        except ValueError as e:
            label = -1
        return label

    def __getitem__(self, index):
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
            val = 0.0  # 默认值
            aro = 0.0  # 默认值
        return (image1, image2), label, val, aro

    def __len__(self):
        return len(self.imgs)

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], '.2f') if normalize else str(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def test():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet_Smirk_Spatial_Alignment(num_class=args.num_class, num_head=args.num_head)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 定义数据变换和数据加载器
    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = CustomImageFolder(args.root1 + '/val', args.root2 + '/val', args.val_excel_path, transform=data_transforms_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_predicted = []

    with torch.no_grad():
        for imgs, targets, aro_true, val_true in tqdm(val_loader):
            image1, image2 = imgs
            image1, image2 = image1.to(device), image2.to(device)
            expression_true = targets.to(device)
            pred_expression, _, _, pred_aro, pred_val, _ = model(image1, image2)

            _, predicted = torch.max(pred_expression, 1)
            all_targets.extend(expression_true.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

            total_correct += (predicted == expression_true).sum().item()
            total_samples += expression_true.size(0)

    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']
    matrix = confusion_matrix(all_targets, all_predicted)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=class8_names, normalize=True, title='Linear_fusion_Confusion Matrix (acc: %0.2f%%)' % (accuracy * 100))
    os.makedirs('FusionConfusionMatrix', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join('FusionConfusionMatrix', f"Linear_Fusion.png"))
    plt.close()
    precision = precision_score(all_targets, all_predicted, average='weighted')
    recall = recall_score(all_targets, all_predicted, average='weighted')
    f1 = f1_score(all_targets, all_predicted, average='weighted')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
      
if __name__ == "__main__":
    test()
