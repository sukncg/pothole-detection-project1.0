# pothole-detection-project/main.py

import os
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# 设置随机种子确保结果可复现
torch.manual_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)


class PotholeDetectionProject:
    def __init__(self, config_path='configs/config.yaml'):
        """初始化坑洼检测项目"""
        # 加载配置
        with open(config_path, 'rb') as f:      # 使用二进制模式读取
            content = f.read().decode('utf-8')  # 手动解码为UTF-8
            self.config = yaml.safe_load(content)

        # 创建结果目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(self.config['results']['save_dir'], self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # 初始化模型
        self.model = None

    def prepare_data(self):
        """准备数据集"""
        print("Preparing dataset...")
        # 数据配置已经在data.yaml中定义，无需额外处理

    def train_model(self):
        """训练模型"""
        print("Training model...")
        # 创建模型
        if self.config['model']['pretrained']:
            self.model = YOLO(self.config['model']['architecture'] + '.pt')
        else:
            self.model = YOLO(self.config['model']['architecture'])

        # 训练模型
        results = self.model.train(
            data=self.config['data']['yaml_path'],
            epochs=self.config['training']['epochs'],
            imgsz=self.config['data']['image_size'],
            batch=self.config['training']['batch_size'],
            lr0=self.config['training']['learning_rate'],
            device=self.device.type,
            project=self.results_dir,
            name='training',
            pretrained=self.config['model']['pretrained'],
            patience=self.config['training']['early_stopping_patience']
        )

        # 保存模型
        best_model_path = os.path.join(self.results_dir, 'training', 'weights', 'best.pt')
        self.model = YOLO(best_model_path)
        print(f"Model saved to: {best_model_path}")

        return results

    def evaluate_model(self):
        """评估模型性能"""
        print("Evaluating model...")
        # 在验证集上评估
        metrics = self.model.val()

        # 保存评估结果
        metrics_path = os.path.join(self.results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            f.write(str(metrics))

        print(f"Evaluation metrics saved to: {metrics_path}")
        return metrics

    def predict(self, image_path=None, image_dir=None):
        """使用模型进行预测"""
        print("Making predictions...")
        if image_path:
            # 预测单张图像
            results = self.model.predict(source=image_path, save=True, project=self.results_dir, name='predictions')
        elif image_dir:
            # 预测目录中的所有图像
            results = self.model.predict(source=image_dir, save=True, project=self.results_dir, name='predictions')
        else:
            # 默认预测测试集
            #test_images_dir = os.path.join(
            #    os.path.dirname(os.path.dirname(self.config['data']['test_dir'])),  # 数据集测试集目录
            #    'images'
            #)
            test_images_dir = "./dataset/test/images"

            results = self.model.predict(source=test_images_dir, save=True, project=self.results_dir,
                                         name='predictions')

        return results

    def analyze_results(self, metrics, class_names=['severe', 'moderate', 'mild']):
        """分析并可视化结果"""
        print("Analyzing results...")

        # 1. 绘制混淆矩阵
        conf_matrix = metrics.confusion_matrix.matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()

        # 2. 绘制PR曲线
        plt.figure(figsize=(12, 10))
        for i, class_name in enumerate(class_names):
            precision = metrics.pr_curve[i][0].tolist()
            recall = metrics.pr_curve[i][1].tolist()
            plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {metrics.ap_class_index[i]:.3f})')

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'pr_curve.png'))
        plt.close()

        # 3. 绘制训练历史
        if hasattr(metrics, 'results_file') and os.path.exists(metrics.results_file):
            results_df = pd.read_csv(metrics.results_file)

            # 绘制损失曲线
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(results_df['train/box_loss'], label='Box Loss')
            plt.plot(results_df['train/obj_loss'], label='Object Loss')
            plt.plot(results_df['train/cls_loss'], label='Class Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)

            # 绘制mAP曲线
            plt.subplot(1, 2, 2)
            plt.plot(results_df['metrics/mAP50(B)'], label='mAP@0.5')
            plt.plot(results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Validation mAP')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'training_history.png'))
            plt.close()

        # 4. 保存类别性能指标
        metrics_dict = {
            'Class': class_names,
            'Precision': metrics.p[:, 0].tolist(),
            'Recall': metrics.r[:, 0].tolist(),
            'F1-Score': 2 * metrics.p[:, 0] * metrics.r[:, 0] / (metrics.p[:, 0] + metrics.r[:, 0] + 1e-16),
            'mAP@0.5': metrics.ap50[:, 0].tolist(),
            'mAP@0.5:0.95': metrics.ap[:, 0].tolist()
        }

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_csv(os.path.join(self.results_dir, 'class_metrics.csv'), index=False)

        print(f"Results analysis saved to: {self.results_dir}")

    def run_full_pipeline(self):
        """运行完整的项目流程"""
        self.prepare_data()
        results = self.train_model()
        metrics = self.evaluate_model()
        self.predict()
        self.analyze_results(metrics)
        print(f"Project completed. All results saved to: {self.results_dir}")


if __name__ == "__main__":
    # 创建并运行项目
    project = PotholeDetectionProject()
    project.run_full_pipeline()