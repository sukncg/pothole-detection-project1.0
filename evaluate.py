# pothole-detection-project/evaluate.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml
import os
import numpy as np
import pandas as pd
from models.pothole_detector import PotholeDetector
from utils.dataset import PotholeDataset
from utils.train_utils import evaluate_model, calculate_class_metrics
from utils.visualization import visualize_predictions, plot_confusion_matrix, plot_pr_curve


def main():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['val_transforms'][2]['params']['mean'],
            std=config['data']['val_transforms'][2]['params']['std']
        )
    ])

    # 创建数据集
    test_dataset = PotholeDataset(
        image_dir=os.path.join(config['data']['test_dir'], 'images'),
        label_dir=os.path.join(config['data']['test_dir'], 'labels'),
        transform=test_transform
    )

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 初始化模型并加载预训练权重
    model = PotholeDetector(num_classes=config['model']['num_classes'] + 1, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(config['results']['save_dir'], config['results']['model_save_path'])))
    model = model.to(device)

    # 在测试集上评估
    test_loss, test_mAP = evaluate_model(model, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test mAP: {test_mAP:.4f}')

    # 计算各类别的性能指标
    precision, recall, f1, cm = calculate_class_metrics(model, test_loader, device)

    # 保存类别性能指标
    metrics_df = pd.DataFrame({
        'Class': test_dataset.classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    metrics_path = os.path.join(config['results']['save_dir'], config['results']['class_metrics_path'])
    metrics_df.to_csv(metrics_path, index=False)
    print(f'Class metrics saved to {metrics_path}')

    # 可视化预测结果
    visualize_predictions(
        model,
        test_loader,
        device,
        num_samples=10,
        save_dir=os.path.join(config['results']['save_dir'], config['results']['predictions_dir'])
    )

    # 绘制混淆矩阵
    plot_confusion_matrix(
        cm,
        test_dataset.classes,
        save_path=os.path.join(config['results']['save_dir'], config['results']['confusion_matrix_path'])
    )

    # 绘制PR曲线 (简化版)
    # 注意：这里使用的是所有类别的平均精确率和召回率
    # 实际应用中应分别为每个类别绘制PR曲线
    plot_pr_curve(
        precision,
        recall,
        save_path=os.path.join(config['results']['save_dir'], config['results']['pr_curve_path'])
    )


if __name__ == '__main__':
    main()