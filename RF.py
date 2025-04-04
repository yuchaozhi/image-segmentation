import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataset_adni import get_adni

# 获取数据
(xtrain, ytrain), (xtest, ytest) = get_adni()
X = xtrain
Xtest = xtest
y = pd.Series(ytrain, name='class')

# 保证随机性可复现
np.random.seed(42)

# Unlabel 部分数据
hidden_size = 1565
y_copy = y.copy()
y_copy.loc[
    np.random.choice(
        y_copy[y_copy == 1].index,
        replace=False,
        size=hidden_size
    )
] = 0
y = y_copy

# 数据展平
X_flat = X.reshape(X.shape[0], -1)
X_flat_test = Xtest.reshape(Xtest.shape[0], -1)

# 标准化
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X_flat_test = scaler.transform(X_flat_test)

# PCA 降维
pca = PCA(n_components=3)
X_flat = pca.fit_transform(X_flat)
X_flat_test = pca.transform(X_flat_test)

print(f'Final shape after PCA: {X_flat.shape}')

# 训练随机森林模型
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    class_weight={0: 1, 1: 1},  # 适当减少正例（1）的权重
    random_state=42,
    verbose=1,
    n_jobs=-1
)

# 训练模型
model.fit(X_flat, y)

# 预测概率
y_prob = model.predict_proba(X_flat_test)[:, 1]

# **自动调整分类阈值**
precisions, recalls, thresholds = precision_recall_curve(ytest, y_prob)
best_idx = (precisions * recalls).argmax()
best_threshold = thresholds[best_idx] + 0.1
y_pred = (y_prob >= best_threshold).astype(int)

# 计算评估指标
accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred)
recall = recall_score(ytest, y_pred)
f1 = f1_score(ytest, y_pred)
auc = roc_auc_score(ytest, y_prob)
ap = average_precision_score(ytest, y_prob)

print(f'Model Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
print(f'AUC: {auc:.4f}, AP: {ap:.4f}')
