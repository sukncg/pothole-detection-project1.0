# pothole-detection-project/models/pothole_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络，用于多尺度特征融合"""
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[-2:]
            last_inner = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = last_inner + inner_lateral
            results.insert(0, self.layer_blocks[i](last_inner))
        
        return tuple(results)

class PotholeDetector(nn.Module):
    """基于多尺度特征融合的坑洼检测网络"""
    def __init__(self, num_classes=3, pretrained=True):
        super(PotholeDetector, self).__init__()
        
        # 使用ResNet50作为骨干网络
        resnet = models.resnet50(pretrained=pretrained)
        
        # 提取特征层
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # C2
            resnet.layer2,  # C3
            resnet.layer3,  # C4
            resnet.layer4   # C5
        )
        
        # 特征金字塔网络
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # 检测头
        self.roi_heads = self._create_roi_heads(num_classes)
        
    def _create_roi_heads(self, num_classes):
        """创建区域检测头"""
        # 这里简化实现，实际应用中应使用完整的RoIHeads
        class SimpleRoIHeads(nn.Module):
            def __init__(self, num_classes):
                super(SimpleRoIHeads, self).__init__()
                self.box_head = nn.Sequential(
                    nn.Linear(256 * 7 * 7, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU()
                )
                self.box_predictor = nn.Sequential(
                    nn.Linear(1024, num_classes),  # 类别预测
                    nn.Linear(1024, num_classes * 4)  # 边界框回归
                )
                
            def forward(self, features, images, targets=None):
                # 简化的前向传播
                boxes = []
                labels = []
                scores = []
                
                # 在实际应用中，这里应该实现完整的RoI池化和检测逻辑
                # 为简化示例，我们返回随机预测结果
                for img in images:
                    img_size = img.shape[-2:]
                    n_boxes = torch.randint(1, 5, (1,)).item()
                    
                    # 随机生成边界框
                    box = torch.rand(n_boxes, 4)
                    box[:, 2:] = box[:, :2] + 0.2 * torch.rand(n_boxes, 2)
                    box[:, 2:] = torch.clamp(box[:, 2:], max=0.9)
                    
                    # 随机生成标签和分数
                    label = torch.randint(0, num_classes, (n_boxes,))
                    score = torch.rand(n_boxes)
                    
                    # 转换为图像尺寸
                    box[:, 0::2] *= img_size[1]
                    box[:, 1::2] *= img_size[0]
                    
                    boxes.append(box)
                    labels.append(label)
                    scores.append(score)
                
                detections = []
                for b, l, s in zip(boxes, labels, scores):
                    detections.append({
                        'boxes': b,
                        'labels': l,
                        'scores': s
                    })
                
                # 训练时返回损失
                losses = {}
                if targets is not None:
                    # 实际应用中应实现损失计算
                    losses = {
                        'loss_classifier': torch.tensor(0.0, requires_grad=True),
                        'loss_box_reg': torch.tensor(0.0, requires_grad=True)
                    }
                
                return detections, losses
                
        return SimpleRoIHeads(num_classes)
        
    def forward(self, images, targets=None):
        # 提取特征
        features = self.backbone(images)
        
        # 特征金字塔处理
        features = self.fpn(features)
        
        # 检测
        detections, detector_losses = self.roi_heads(features, images, targets)
        
        if self.training:
            return detector_losses
        else:
            return detections