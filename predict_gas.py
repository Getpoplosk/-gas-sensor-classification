"""
气体传感器信号灰度图像分类预测脚本
使用训练好的模型进行预测
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 气体类别名称（按字母顺序，与训练时一致）
GAS_CLASSES = [
    'Acetaldehyde', 'Acetone', 'Ammonia', 'Benzene', 'Butanol',
    'CO', 'Ethylene', 'Methane', 'Methanol', 'Toluene'
]

# 配置参数
CONFIG = {
    'model_path': 'best_model.pth',
    'image_size': 224,
    'resize_size': 256,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# 预处理变换（与训练时一致）
test_transform = transforms.Compose([
    transforms.Resize((CONFIG['resize_size'], CONFIG['resize_size'])),
    transforms.CenterCrop((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes=10):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")

    # 构建与训练时相同的模型架构
    model = models.resnet18(weights=None)  # 不使用预训练权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG['device'])
    model.eval()

    print(f"模型加载成功！验证准确率: {checkpoint.get('valid_acc', 'N/A'):.2f}%")
    return model

def predict_image(model, image_path):
    """对单张图像进行预测"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 预处理
    image_tensor = test_transform(image).unsqueeze(0).to(CONFIG['device'])

    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = GAS_CLASSES[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score, probabilities.cpu().numpy()[0]

def predict_folder(model, folder_path):
    """对文件夹中的所有图像进行预测"""
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            try:
                predicted_class, confidence, _ = predict_image(model, image_path)
                results.append({
                    'filename': filename,
                    'predicted_class': predicted_class,
                    'confidence': confidence
                })
                print(f"{filename}: {predicted_class} (置信度: {confidence:.2%})")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    return results

def main():
    """主函数 - 示例用法"""
    print("=" * 50)
    print("气体传感器信号灰度图像分类预测")
    print("=" * 50)

    # 1. 加载模型
    model = load_model(CONFIG['model_path'])

    # 2. 示例1: 预测单张图像
    print("\n示例1: 预测单张图像")
    print("-" * 30)

    # 替换为你的图像路径
    test_image_path = "TBT/My/image1.jpg"  # 示例路径

    if os.path.exists(test_image_path):
        predicted_class, confidence, probabilities = predict_image(model, test_image_path)
        print(f"图像: {test_image_path}")
        print(f"预测类别: {predicted_class}")
        print(f"置信度: {confidence:.2%}")
        print(f"所有类别概率:")
        for i, (gas_name, prob) in enumerate(zip(GAS_CLASSES, probabilities)):
            print(f"  {gas_name}: {prob:.2%}")
    else:
        print(f"图像文件不存在: {test_image_path}")
        print("请修改 test_image_path 为你的图像路径")

    # 3. 示例2: 预测文件夹中的所有图像
    print("\n示例2: 预测文件夹中的所有图像")
    print("-" * 30)

    test_folder = "TBT/My"  # 示例文件夹路径

    if os.path.exists(test_folder):
        results = predict_folder(model, test_folder)
        print(f"\n共预测 {len(results)} 张图像")
    else:
        print(f"文件夹不存在: {test_folder}")
        print("请修改 test_folder 为你的文件夹路径")

    # 4. 示例3: 批量预测多个文件夹
    print("\n示例3: 批量预测多个文件夹")
    print("-" * 30)

    test_folders = [
        "TBT/My",
        "TBT/AnotherFolder",
        "TBT/YetAnotherFolder"
    ]

    for folder in test_folders:
        if os.path.exists(folder):
            print(f"\n预测文件夹: {folder}")
            results = predict_folder(model, folder)

    print("\n" + "=" * 50)
    print("预测完成！")
    print("=" * 50)

if __name__ == '__main__':
    main()
