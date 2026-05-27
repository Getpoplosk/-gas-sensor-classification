"""
在其他Python代码中调用训练好的气体分类模型示例
"""

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 气体类别名称
GAS_CLASSES = [
    'Acetaldehyde', 'Acetone', 'Ammonia', 'Benzene', 'Butanol',
    'CO', 'Ethylene', 'Methane', 'Methanol', 'Toluene'
]

class GasClassifier:
    """气体分类器类 - 封装模型加载和预测功能"""

    def __init__(self, model_path='best_model.pth', device=None):
        """
        初始化气体分类器

        Args:
            model_path: 模型文件路径
            device: 计算设备 (cuda/cpu)，如果为None则自动选择
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 预处理变换（与训练时一致）
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载模型
        self.model = self._load_model(model_path)
        print(f"气体分类器已初始化，使用设备: {self.device}")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        # 构建与训练时相同的模型架构
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(GAS_CLASSES))

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def predict(self, image_path):
        """
        预测单张图像

        Args:
            image_path: 图像文件路径

        Returns:
            dict: 包含预测结果的字典
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = GAS_CLASSES[predicted.item()]
        confidence_score = confidence.item()

        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'probabilities': probabilities.cpu().numpy()[0],
            'all_classes': GAS_CLASSES
        }

    def predict_batch(self, image_paths):
        """
        批量预测多张图像

        Args:
            image_paths: 图像文件路径列表

        Returns:
            list: 每个图像的预测结果字典列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"预测 {image_path} 时出错: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results


# 使用示例
if __name__ == '__main__':
    # 1. 初始化分类器
    classifier = GasClassifier(model_path='best_model.pth')

    # 2. 预测单张图像
    print("\n示例1: 预测单张图像")
    image_path = "test/Acetaldehyde/001.png"  # 替换为你的图像路径
    if os.path.exists(image_path):
        result = classifier.predict(image_path)
        print(f"图像: {image_path}")
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.2%}")
    else:
        print(f"图像文件不存在: {image_path}")

    # 3. 批量预测
    print("\n示例2: 批量预测")
    image_paths = [
        "test/Acetaldehyde/001.png",
        "test/Acetone/001.png",
        "test/Ammonia/001.png"
    ]
    results = classifier.predict_batch(image_paths)
    for r in results:
        if 'error' not in r:
            print(f"{r['image_path']}: {r['predicted_class']} ({r['confidence']:.2%})")
