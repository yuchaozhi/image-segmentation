import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


# 定义样本数量和特征数量
num_samples = 4
num_features = 6

# 随机生成两组数据
data_top = np.random.rand(num_samples, num_features)
data_bottom = np.random.rand(num_samples, num_features)

# 配对 t 检验
t_statistic, p_values = ttest_rel(data_top, data_bottom)

# 显著性判断（95% 置信度）
alpha = 0.05
significance = p_values < alpha

# 打印检验结果
print("Column | t-stat | p-value | Significant (95%)")
for i in range(num_features):
    sig = "Yes" if significance[i] else "No"
    print(f"   {i+1}    | {t_statistic[i]:.4f} | {p_values[i]:.4f} | {sig}")

# ---------- 可选：绘图展示 ----------
labels = [f'Metric {i+1}' for i in range(num_features)]
top_means = data_top.mean(axis=0)
bottom_means = data_bottom.mean(axis=0)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, top_means, width, label='Top', color='skyblue')
bars2 = ax.bar(x + width/2, bottom_means, width, label='Bottom', color='salmon')

# 显著性星号标注
for i in range(len(labels)):
    if significance[i]:
        max_height = max(top_means[i], bottom_means[i])
        ax.text(x[i], max_height + 0.01, '*', ha='center', va='bottom', fontsize=14, color='red')

# 设置图表信息
ax.set_ylabel('Mean Value')
ax.set_title('Paired t-test Results (95% Confidence)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()
