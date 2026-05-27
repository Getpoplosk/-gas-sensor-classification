"""
气体传感器信号灰度图像分类训练脚本
数据集：10种不同气体的传感器信号灰度图像
模型：ResNet18预训练模型
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ==================== 配置参数 ====================
CONFIG = {
    'data_dir': '.',  # 数据根目录
    'train_dir': 'train',  # 训练集目录
    'valid_dir': 'valid',  # 验证集目录
    'test_dir': 'test',    # 测试集目录
    'batch_size': 32,
    'num_epochs': 20,  # 减少epoch数量
    'learning_rate': 0.0005,  # 降低学习率
    'num_classes': 10,  # 10种气体
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # 自动选择GPU或CPU
    'image_size': 224,  # ResNet输入尺寸
    'resize_size': 256,  # Resize尺寸
    'patience': 5,  # 早停耐心值
    'min_delta': 0.001,  # 最小改进阈值
}

# 气体类别名称（按字母顺序）
GAS_CLASSES = [
    'Acetaldehyde', 'Acetone', 'Ammonia', 'Benzene', 'Butanol',
    'CO', 'Ethylene', 'Methane', 'Methanol', 'Toluene'
]

print(f"使用设备: {CONFIG['device']}")
print(f"训练集路径: {CONFIG['train_dir']}")
print(f"验证集路径: {CONFIG['valid_dir']}")
print(f"测试集路径: {CONFIG['test_dir']}")
print(f"批次大小: {CONFIG['batch_size']}")
print(f"训练轮数: {CONFIG['num_epochs']}")
print(f"学习率: {CONFIG['learning_rate']}")
print(f"类别数量: {CONFIG['num_classes']}")
print("-" * 50)


# ==================== 数据预处理 ====================
# 定义数据增强和预处理
# 训练集：随机裁剪、水平翻转、颜色抖动等增强
train_transform = transforms.Compose([
    # Resize到256x256
    transforms.Resize((CONFIG['resize_size'], CONFIG['resize_size'])),
    # 随机裁剪到224x224
    transforms.RandomCrop((CONFIG['image_size'], CONFIG['image_size'])),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(p=0.5),
    # 随机旋转（±15度）
    transforms.RandomRotation(15),
    # 颜色抖动（亮度、对比度、饱和度）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 转换为Tensor
    transforms.ToTensor(),
    # 标准化（ImageNet统计量）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集和测试集：仅Resize和中心裁剪
test_transform = transforms.Compose([
    # Resize到256x256
    transforms.Resize((CONFIG['resize_size'], CONFIG['resize_size'])),
    # 中心裁剪到224x224
    transforms.CenterCrop((CONFIG['image_size'], CONFIG['image_size'])),
    # 转换为Tensor
    transforms.ToTensor(),
    # 标准化（ImageNet统计量）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==================== 数据加载 ====================
def load_datasets():
    """使用ImageFolder加载数据集"""
    print("正在加载数据集...")

    # 训练集
    train_dataset = datasets.ImageFolder(
        root=os.path.join(CONFIG['data_dir'], CONFIG['train_dir']),
        transform=train_transform
    )

    # 验证集
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(CONFIG['data_dir'], CONFIG['valid_dir']),
        transform=test_transform
    )

    # 测试集
    test_dataset = datasets.ImageFolder(
        root=os.path.join(CONFIG['data_dir'], CONFIG['test_dir']),
        transform=test_transform
    )

    print(f"训练集大小: {len(train_dataset)} 张图像")
    print(f"验证集大小: {len(valid_dataset)} 张图像")
    print(f"测试集大小: {len(test_dataset)} 张图像")
    print(f"类别映射: {train_dataset.class_to_idx}")
    print("-" * 50)

    return train_dataset, valid_dataset, test_dataset


def create_dataloaders(train_dataset, valid_dataset, test_dataset):
    """创建数据加载器"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader


# ==================== 模型构建 ====================
def build_model(num_classes=10):
    """构建ResNet18预训练模型"""
    print("正在构建ResNet18模型...")

    # 加载预训练的ResNet18模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 冻结前面的卷积层（可选，用于微调）
    # for param in model.parameters():
    #     param.requires_grad = False

    # 修改全连接层以适应10类分类任务
    # ResNet18的fc层输入特征数是512
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 将模型移动到指定设备
    model = model.to(CONFIG['device'])

    print(f"模型参数总量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 50)

    return model


# ==================== 训练函数 ====================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]} [Train]')

    for inputs, labels in pbar:
        # 将数据移动到设备
        inputs = inputs.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条描述
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


# ==================== 验证函数 ====================
def validate(model, data_loader, criterion, phase='Validation'):
    """验证或测试模型"""
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度以节省内存
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f'{phase}'):
            # 将数据移动到设备
            inputs = inputs.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


# ==================== 主训练循环 ====================
def train_model(model, train_loader, valid_loader, criterion, optimizer, start_epoch=0, best_valid_acc=0.0):
    """主训练循环"""
    print(f"从第 {start_epoch + 1} 个epoch开始训练...")
    print(f"配置: epochs={CONFIG['num_epochs']}, lr={CONFIG['learning_rate']}, patience={CONFIG['patience']}")
    print("-" * 50)

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }

    best_model_path = 'best_model.pth'
    patience_counter = 0  # 早停计数器

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        # 验证
        valid_loss, valid_acc = validate(model, valid_loader, criterion, 'Validation')

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        # 计算耗时
        epoch_time = time.time() - start_time

        # 打印结果
        print(f'Epoch {epoch+1}/{CONFIG["num_epochs"]} | '
              f'Time: {epoch_time:.1f}s | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Valid Loss: {valid_loss:.4f} | '
              f'Valid Acc: {valid_acc:.2f}%')

        # 保存最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
                'valid_loss': valid_loss
            }, best_model_path)
            print(f'>>> 保存最佳模型 (Valid Acc: {valid_acc:.2f}%)')
            patience_counter = 0  # 重置早停计数器
        else:
            # 检查是否满足早停条件
            if valid_acc < best_valid_acc - CONFIG['min_delta']:
                patience_counter += 1
                print(f'>>> 早停计数器: {patience_counter}/{CONFIG["patience"]}')

                if patience_counter >= CONFIG['patience']:
                    print(f'>>> 早停触发！验证准确率连续{CONFIG["patience"]}个epoch未提升')
                    print(f'>>> 最佳验证准确率: {best_valid_acc:.2f}%')
                    break

        print('-' * 50)

    print(f'训练完成！最佳验证准确率: {best_valid_acc:.2f}%')
    return history, best_model_path


# ==================== 绘制训练曲线 ====================
def plot_training_history(history, save_path='曲线图.png'):
    """绘制训练和验证的Loss、Accuracy曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['valid_loss'], 'r-', label='Valid Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['valid_acc'], 'r-', label='Valid Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存至: {save_path}")
    plt.close()


# ==================== 绘制混淆矩阵 ====================
def plot_confusion_matrix(model, test_loader, save_path='混淆矩阵.png'):
    """绘制测试集的混淆矩阵热力图"""
    print("\n正在计算混淆矩阵...")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='测试集预测'):
            inputs = inputs.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=GAS_CLASSES,
        yticklabels=GAS_CLASSES,
        cbar_kws={'label': '样本数量'}
    )
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title('测试集混淆矩阵', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")
    plt.close()

    # 计算并打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=GAS_CLASSES, digits=4))

    return cm


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 50)
    print("气体传感器信号灰度图像分类训练脚本")
    print("=" * 50)

    # 1. 加载数据集
    train_dataset, valid_dataset, test_dataset = load_datasets()

    # 2. 创建数据加载器
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset, valid_dataset, test_dataset
    )

    # 3. 构建模型
    model = build_model(num_classes=CONFIG['num_classes'])

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 5. 检查是否有检查点可以继续训练
    checkpoint_path = 'best_model.pth'
    start_epoch = 0
    best_valid_acc = 0.0

    if os.path.exists(checkpoint_path):
        print(f"\n发现检查点文件: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 1)  # 已训练的epoch数
            best_valid_acc = checkpoint.get('valid_acc', 0.0)

            # 检查是否已经完成训练
            if start_epoch >= CONFIG['num_epochs']:
                print(f"模型已训练完成 (Epoch {start_epoch}/{CONFIG['num_epochs']})")
                print(f"验证准确率: {best_valid_acc:.2f}%")
                print("将从头开始训练")
                start_epoch = 0
                best_valid_acc = 0.0
            else:
                print(f"已加载模型状态 - Epoch: {start_epoch}, Valid Acc: {best_valid_acc:.2f}%")
                print(f"将从第 {start_epoch + 1} 个epoch开始继续训练")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("将从头开始训练")
    else:
        print("\n未发现检查点文件，将从头开始训练")

    # 6. 训练模型
    history, best_model_path = train_model(
        model, train_loader, valid_loader, criterion, optimizer, start_epoch, best_valid_acc
    )

    # 7. 绘制训练曲线
    plot_training_history(history, save_path='曲线图.png')

    # 8. 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(best_model_path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG['device'])

    # 9. 在测试集上评估
    test_loss, test_acc = validate(model, test_loader, criterion, 'Test')
    print(f"\n测试集结果 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    # 10. 绘制混淆矩阵
    plot_confusion_matrix(model, test_loader, save_path='混淆矩阵.png')

    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()
