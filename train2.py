import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

def main():
    # 设置 CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(224),  # 将最小边调整到 224
        transforms.CenterCrop(224),  # 从中心裁剪出 224x224 的区域
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 数据集路径
    dataset_path = 'E:\\TrainPicture\\Plant Disease Classification Merged Dataset'

    # 加载数据集
    dataset = ImageFolder(root=dataset_path, transform=transform)

    # 划分数据集为训练集和验证集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 加载预训练的 ResNet18 模型
    net = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 修改最后的全连接层以匹配类别数
    num_ftrs = net.fc.in_features
    num_classes = len(dataset.classes)
    net.fc = nn.Linear(num_ftrs, num_classes)

    # 将模型移至CUDA
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 用于绘制学习曲线的列表
    train_losses = []
    val_accuracies = []

    try:
        for epoch in range(100):  # 训练轮次
            running_loss = 0.0
            for i, batch_data in enumerate(train_loader, 0):
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            train_losses.append(running_loss / len(train_loader))

            # 验证阶段
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data in test_loader:
                    images, labels = batch_data[0].to(device), batch_data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)
            net.train()
            print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")

    except KeyboardInterrupt:
        print("训练被中断，保存当前模型...")

    finally:
        # 保存模型
        save_path = 'E:\\ModelWeights\\plant_disease_model_interrupted.pth'
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(net.state_dict(), save_path)
        print(f"模型已保存至 {save_path}")

        # 再次评估模型以得到最终的测试准确率
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in test_loader:
                images, labels = batch_data[0].to(device), batch_data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        final_val_accuracy = 100 * correct / total
        print('Final Accuracy of the network on the test images: %d %%' % final_val_accuracy)

        # 绘制学习曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    main()
