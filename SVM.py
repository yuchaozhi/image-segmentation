import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from dataset_adni import get_adni
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV

# 获取数据
(xtrain, ytrain), (xtest, ytest) = get_adni()
X = xtrain
Xtest = xtest
y = ytrain
y = pd.Series(y, name='class')

# Unlabel a certain number of data points
hidden_size = 1565  # 1565
y.loc[
    np.random.choice(
        y[y == 1].index,
        replace=False,
        size=hidden_size
    )
] = 0

# 转换数据为 PyTorch Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

X_tensor_test = torch.tensor(Xtest, dtype=torch.float32)
y_tensor_test = torch.tensor(ytest, dtype=torch.long)

# 将数据展平为一维
X_flat = X.reshape(X.shape[0], -1)  # 将每个图像展平成一个一维向量
X_flat_test = Xtest.reshape(Xtest.shape[0], -1)  # 对测试数据进行相同处理
print(X_flat.shape)

# 标准化数据
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X_flat_test = scaler.transform(X_flat_test)

# 使用PCA降维
pca = PCA(n_components=10)  # 将特征维度减少到100，或者你可以使用交叉验证来选择合适的维度
X_flat = pca.fit_transform(X_flat)
X_flat_test = pca.transform(X_flat_test)

# 训练SVM模型
model = LinearSVC(max_iter=10, verbose=True)  # 使用线性SVM

# 使用 CalibratedClassifierCV 估计概率
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
model.fit(X_flat, y)

# 使用已校准的模型来获取预测概率
calibrated_model.fit(X_flat, y)

# 预测标签
y_pred = model.predict(X_flat_test)

# 获取预测的概率（正类的概率）
y_prob = calibrated_model.predict_proba(X_flat_test)[:, 1]

# 计算准确率
accuracy = accuracy_score(ytest, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# 计算精确率、召回率、F1-score
precision = precision_score(ytest, y_pred)
recall = recall_score(ytest, y_pred)
f1 = f1_score(ytest, y_pred)

# 计算ROC曲线面积（AUC）和平均精确率（AP）
auc = roc_auc_score(ytest, y_prob)
ap = average_precision_score(ytest, y_prob)

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
print(f'AUC: {auc:.4f}, AP: {ap:.4f}')
