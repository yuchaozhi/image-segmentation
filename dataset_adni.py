import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset


def get_adni():
    # 1. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(  # 归一化
            mean=[0.2817, 0.2817, 0.2817],  # 均值
            std=[0.3277, 0.3277, 0.3277]  # 标准差
        )
    ])

    # 2. 加载数据集
    data_dir = r'AD/Alzheimer_MRI_4_classes_dataset'  # 替换为你的数据集路径
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 3. 将数据集分为健康和痴呆两类
    # 定义健康和痴呆的类别
    healthy_class = 'NonDemented'
    dementia_classes = ['MildDemented', 'ModerateDemented', 'VeryMildDemented']

    # 3. 构建二分类数据集包装器
    class BinaryDementiaDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.healthy_class = healthy_class

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            # image=image[0, :, :]
            class_name = self.dataset.classes[label]
            binary_label = 0 if class_name == self.healthy_class else 1
            return image, binary_label

    # 4. 包装数据集
    dataset = BinaryDementiaDataset(dataset)

    # 更新 class_to_idx 和 classes
    dataset.class_to_idx = {healthy_class: 0, 'Dementia': 1}
    dataset.classes = [healthy_class, 'Dementia']

    # 4. 划分训练集和测试集（8:2）
    train_ratio = 0.8
    test_ratio = 0.2
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 5. 创建 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义提取数据的函数
    def extract_data_from_loader(loader):
        x_all = []
        y_all = []
        for x, y in loader:
            x_all.append(x)
            y_all.append(y)
        # 将批次数据拼接起来
        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        # 转换为 NumPy 数组
        x_all = x_all.numpy()
        y_all = y_all.numpy()
        return x_all, y_all

    xtrain, ytrain = extract_data_from_loader(train_loader)
    xtest, ytest = extract_data_from_loader(test_loader)

    return (xtrain,ytrain),(xtest,ytest)
