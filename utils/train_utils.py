# pothole-detection-project/utils/train_utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, save_path='best_model.pth'):
    """训练模型"""
    best_mAP = 0.0
    history = {'train_loss': [], 'val_loss': [], 'mAP': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            losses.backward()
            optimizer.step()

            running_loss += losses.item() * len(images)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')

        # 验证阶段
        val_loss, mAP = evaluate_model(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['mAP'].append(mAP)
        print(f'Val Loss: {val_loss:.4f} mAP: {mAP:.4f}')

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with mAP: {mAP:.4f}')

    return history


def evaluate_model(model, val_loader, device):
    """评估模型性能"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item() * len(images)

            # 收集预测结果
            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                for box, label, score in zip(boxes, labels, scores):
                    all_predictions.append({
                        'image_id': targets[i]['image_id'].item(),
                        'category_id': label,
                        'bbox': box.tolist(),
                        'score': score
                    })

                # 收集真实目标
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()

                for box, label in zip(gt_boxes, gt_labels):
                    all_targets.append({
                        'image_id': targets[i]['image_id'].item(),
                        'category_id': label,
                        'bbox': box.tolist(),
                        'iscrowd': 0
                    })

    # 计算平均精度均值(mAP)
    mAP = calculate_mAP(all_predictions, all_targets)

    return running_loss / len(val_loader.dataset), mAP


def calculate_mAP(predictions, targets, iou_threshold=0.5):
    """计算平均精度均值(mAP)"""
    # 这里使用简化版计算，实际应用中应使用COCO评估工具
    # 提取每个类别的预测和目标
    categories = [0, 1, 2]  # 假设类别ID为0,1,2
    aps = []

    for cat_id in categories:
        cat_preds = [p for p in predictions if p['category_id'] == cat_id]
        cat_targets = [t for t in targets if t['category_id'] == cat_id]

        # 按置信度排序
        cat_preds.sort(key=lambda x: x['score'], reverse=True)

        # 计算每个预测的TP和FP
        tp = np.zeros(len(cat_preds))
        fp = np.zeros(len(cat_preds))
        matched_targets = set()

        for pred_idx, pred in enumerate(cat_preds):
            max_iou = 0
            best_target_idx = -1

            for target_idx, target in enumerate(cat_targets):
                if target['image_id'] != pred['image_id']:
                    continue

                iou = calculate_iou(pred['bbox'], target['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_target_idx = target_idx

            if max_iou >= iou_threshold and best_target_idx not in matched_targets:
                tp[pred_idx] = 1
                matched_targets.add(best_target_idx)
            else:
                fp[pred_idx] = 1

        # 计算累积TP和FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        # 计算精确率和召回率
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / max(len(cat_targets), 1)

        # 计算AP (使用11点插值法)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0

        aps.append(ap)

    # 计算mAP
    mAP = np.mean(aps) if aps else 0
    return mAP


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # 转换为[x_min, y_min, x_max, y_max]格式
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集区域
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # 计算交集面积
    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)

    # 计算并集面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def calculate_class_metrics(model, test_loader, device):
    """计算各类别的精确率、召回率和F1分数"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_labels = output['labels'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()

                # 确保预测和真实标签数量一致
                if len(pred_labels) > len(gt_labels):
                    pred_labels = pred_labels[:len(gt_labels)]
                elif len(pred_labels) < len(gt_labels):
                    pred_labels = np.concatenate([pred_labels, np.full(len(gt_labels) - len(pred_labels), -1)])

                all_preds.extend(pred_labels)
                all_labels.extend(gt_labels)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 计算精确率、召回率和F1分数
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, cm