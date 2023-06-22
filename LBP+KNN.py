import os
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torchvision.models as models
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# 定义数据集路径
dataset_path = 'data_sex'

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像大小调整为 224x224
    transforms.ToTensor(),  # 转换为张量
])

# 加载训练集数据
train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)
# 加载测试集数据
test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)

# 创建特征提取器模型
resnet = models.resnet50()  # 使用ResNet-50作为特征提取器
resnet.load_state_dict(torch.load('resnet50-0676ba61.pth'))  # 加载预训练的权重
resnet = resnet.eval()

# 提取训练集特征向量
train_features = []
train_labels = []

for image, label in train_dataset:
    # 提取图像特征向量
    feature_vector = resnet(image.unsqueeze(0)).detach().numpy().flatten()
    # 提取单个图像的LBP特征
    lbp = local_binary_pattern(image[0].numpy().astype(np.uint8), 24, 3, method='default')
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
    # 将ResNet特征和LBP特征拼接在一起
    feature_vector = np.concatenate((feature_vector, lbp_hist))
    train_features.append(feature_vector)
    train_labels.append(label)

train_features = np.array(train_features)
train_labels = np.array(train_labels)

# 提取测试集特征向量
test_features = []
test_labels = []

for image, label in test_dataset:
    # 提取图像特征向量
    feature_vector = resnet(image.unsqueeze(0)).detach().numpy().flatten()
    # 提取单个图像的LBP特征
    lbp = local_binary_pattern(image[0].numpy().astype(np.uint8), 24, 3, method='default')
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
    # 将ResNet特征和LBP特征拼接在一起
    feature_vector = np.concatenate((feature_vector, lbp_hist))
    test_features.append(feature_vector)
    test_labels.append(label)

test_features = np.array(test_features)
test_labels = np.array(test_labels)

# 定义K折交叉验证
k = 5
kf = KFold(n_splits=k)

# 创建KNN分类器对象
classifier = KNeighborsClassifier(n_neighbors=10)

train_accuracies = []
test_accuracies = []

# 进行K折交叉验证
for train_index, val_index in kf.split(train_features):
    # 划分训练集和验证集
    X_train, X_val = train_features[train_index], train_features[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    # 在训练集上训练分类器
    classifier.fit(X_train, y_train)

    # 在训练集上进行预测
    train_pred = classifier.predict(X_train)

    # 在验证集上进行预测
    val_pred = classifier.predict(X_val)

    # 计算训练集和验证集的准确率
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(val_accuracy)

# 在测试集上进行预测
test_pred = classifier.predict(test_features)

# 计算测试集的准确率
test_accuracy = accuracy_score(test_labels, test_pred)

print("训练集准确率（平均）:", np.mean(train_accuracies))
print("验证集准确率（平均）:", np.mean(test_accuracies))
print("测试集准确率:", test_accuracy)
