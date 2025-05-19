# pothole-detection-project/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from PIL import Image


def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史曲线"""
    plt.figure(figsize=(18, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 绘制mAP曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['mAP'], label='mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_predictions(model, test_loader, device, num_samples=5, save_dir='results/predictions'):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            if i >= num_samples:
                break

            image = images[0].to(device)
            output = model([image])[0]

            # 转换图像用于显示
            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # 创建画布
            plt.figure(figsize=(15, 5))

            # 显示原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title('Original Image')
            plt.axis('off')

            # 显示真实标注
            plt.subplot(1, 3, 2)
            plt.imshow(img_np)
            plt.title('Ground Truth')

            gt_boxes = targets[0]['boxes'].cpu().numpy()
            gt_labels = targets[0]['labels'].cpu().numpy()

            for box, label in zip(gt_boxes, gt_labels):
                x_min, y_min, x_max, y_max = box
                plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                  linewidth=2, edgecolor='g', facecolor='none'))
                plt.gca().text(x_min, y_min, f'{test_loader.dataset.classes[label]}',
                               color='g', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            plt.axis('off')

            # 显示预测结果
            plt.subplot(1, 3, 3)
            plt.imshow(img_np)
            plt.title('Predictions')

            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score > 0.5:  # 只显示置信度大于0.5的预测
                    x_min, y_min, x_max, y_max = box
                    plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                      linewidth=2, edgecolor='r', facecolor='none'))
                    plt.gca().text(x_min, y_min, f'{test_loader.dataset.classes[label]}: {score:.2f}',
                                   color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
            plt.close()


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(precision, recall, save_path='results/pr_curve.png'):
    """绘制精确率-召回率曲线"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()