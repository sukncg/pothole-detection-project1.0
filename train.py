# pothole-detection-project/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml
import os
from models.pothole_detector import PotholeDetector
from utils.dataset import PotholeDataset
from utils.train_utils import train_model, evaluate_model
from utils.visualization import plot_training_history


def main():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建保存目录
    os.makedirs(config['results']['save_dir'], exist_ok=True)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=config['data']['train_transforms'][3]['params']['brightness'],
            contrast=config['data']['train_transforms'][3]['params']['contrast'],
            saturation=config['data']['train_transforms'][3]['params']['saturation'],
            hue=config['data']['train_transforms'][3]['params']['hue']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['train_transforms'][5]['params']['mean'],
            std=config['data']['train_transforms'][5]['params']['std']
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['data']['val_transforms'][2]['params']['mean'],
            std=config['data']['val_transforms'][2]['params']['std']
        )
    ])

    # 创建数据集
    train_dataset = PotholeDataset(
        image_dir=os.path.join(config['data']['train_dir'], 'images'),
        label_dir=os.path.join(config['data']['train_dir'], 'labels'),
        transform=train_transform
    )

    val_dataset = PotholeDataset(
        image_dir=os.path.join(config['data']['val_dir'], 'images'),
        label_dir=os.path.join(config['data']['val_dir'], 'labels'),
        transform=val_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    # 初始化模型
    model = PotholeDetector(num_classes=config['model']['num_classes'] + 1, pretrained=config['model']['pretrained'])
    model = model.to(device)

    # 定义优化器和学习率调度器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['lr_scheduler']['factor'],
        patience=config['training']['lr_scheduler']['patience'],
        verbose=config['training']['lr_scheduler']['verbose']
    )

    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['training']['epochs'],
        device=device,
        save_path=os.path.join(config['results']['save_dir'], config['results']['model_save_path'])
    )

    # 可视化训练历史
    plot_training_history(
        history,
        save_path=os.path.join(config['results']['save_dir'], config['results']['training_history_path'])
    )

    # 加载最佳模型并在验证集上评估
    model.load_state_dict(torch.load(os.path.join(config['results']['save_dir'], config['results']['model_save_path'])))
    val_loss, val_mAP = evaluate_model(model, val_loader, device)
    print(f'Final Validation Loss: {val_loss:.4f}')
    print(f'Final Validation mAP: {val_mAP:.4f}')


if __name__ == '__main__':
    main()