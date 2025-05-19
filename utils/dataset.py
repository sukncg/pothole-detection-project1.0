# pothole-detection-project/utils/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PotholeDataset(Dataset):
    """坑洼检测数据集"""

    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))

        # 类别映射
        self.classes = ['severe', 'moderate', 'mild']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 读取对应的标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    # 格式: [类别, x_center, y_center, width, height]
                    cls = int(line[0])
                    x_center = float(line[1])
                    y_center = float(line[2])
                    width = float(line[3])
                    height = float(line[4])

                    # 转换为[x_min, y_min, x_max, y_max]格式
                    x_min = (x_center - width / 2)
                    y_min = (y_center - height / 2)
                    x_max = (x_center + width / 2)
                    y_max = (y_center + height / 2)

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(cls)

        # 转换为Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target