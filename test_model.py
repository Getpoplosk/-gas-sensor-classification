"""
测试训练好的模型是否能正常加载和预测
"""

import torch
from torchvision import models, transforms
from PIL import Image
import os

# 气体类别名称
GAS_CLASSES = [
    'Acetaldehyde', 'Acetone', 'Ammonia', 'Benzene', 'Butanol',
    'CO', 'Ethylene', 'Methane', 'Methanol', 'Toluene'
]

def test_model():
    """测试模型加载和预测"""
    print("=" * 50)
    print("测试训练好的气体分类模型")
    print("=" * 50)

    # 1. 检查模型文件
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在！")
        return False

    print(f"[OK] 找到模型文件: {model_path}")

    # 2. 加载模型
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 构建模型架构
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(GAS_CLASSES))

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"[OK] 模型加载成功！验证准确率: {checkpoint.get('valid_acc', 'N/A'):.2f}%")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        return False

    # 3. 测试预测功能
    print("\n测试预测功能...")

    # 预处理变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 查找测试图像
    test_folders = ['test/Acetaldehyde', 'test/Acetone', 'test/Ammonia']
    test_image = None

    for folder in test_folders:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image = os.path.join(folder, images[0])
                break

    if test_image:
        try:
            # 加载图像
            image = Image.open(test_image).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # 预测
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = GAS_CLASSES[predicted.item()]
            confidence_score = confidence.item()

            print(f"[OK] 预测成功！")
            print(f"  测试图像: {test_image}")
            print(f"  预测类别: {predicted_class}")
            print(f"  置信度: {confidence_score:.2%}")

            # 显示所有类别的概率
            print(f"\n  所有类别概率:")
            for i, (gas_name, prob) in enumerate(zip(GAS_CLASSES, probabilities.cpu().numpy()[0])):
                print(f"    {gas_name}: {prob:.2%}")

            return True
        except Exception as e:
            print(f"[ERROR] 预测失败: {e}")
            return False
    else:
        print("[WARN] 未找到测试图像，跳过预测测试")
        return True

if __name__ == '__main__':
    success = test_model()
    print("\n" + "=" * 50)
    if success:
        print("[OK] 模型测试通过！可以正常使用。")
    else:
        print("[ERROR] 模型测试失败！请检查模型文件和代码。")
    print("=" * 50)
