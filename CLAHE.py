import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def apply_clahe(image, clip_limit=4.0, grid_size=(8, 8)):
    """ 对图像应用 CLAHE，跳过灰度值为 0 的像素 """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # 仅处理非 0 像素
    l_clahe = np.where(l > 0, clahe.apply(l), l)

    # 合并回 LAB 并转换回 BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_image, l, l_clahe  # 返回原始 L 通道和 CLAHE 后的 L 通道

def plot_histograms(image_paths, save_path):
    """ 绘制四张图像的 CLAHE 处理前后的 L 通道直方图，并添加 a, b, c, d 标注 """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()  # 将 2x2 数组展平，方便索引

    # labels = ["(a)", "(b)", "(c)", "(d)"]  # 子图编号

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is not None:
            _, original_l, clahe_l = apply_clahe(image, clip_limit=4.0, grid_size=(8, 8))

            # 计算直方图（排除 0 像素）
            hist_original = cv2.calcHist([original_l[original_l > 0]], [0], None, [255], [1, 256])
            hist_clahe = cv2.calcHist([clahe_l[clahe_l > 0]], [0], None, [255], [1, 256])

            # 绘制直方图
            axes[i].plot(hist_original, color='blue', label='原始亮度')
            axes[i].plot(hist_clahe, color='red', label='CLAHE亮度通道')
            axes[i].set_xlabel("灰度值")
            axes[i].set_ylabel("像素数量")
            axes[i].legend()

            # # 在每个子图下方添加编号
            # axes[i].text(0.5, -0.15, labels[i], fontsize=14, ha='center', va='center', transform=axes[i].transAxes)

        else:
            axes[i].set_title(f"Error loading Image {i+1}")
            print(f"无法加载图像: {image_path}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=1200)
    plt.show()

# 定义图片路径
image_paths = [
    r"D:\MyCourse\PythonFiles\Paper\STAMT-main\AD\fig\fig\1.jpg",
    r"D:\MyCourse\PythonFiles\Paper\STAMT-main\AD\fig\fig\2.png",
    r"D:\MyCourse\PythonFiles\Paper\STAMT-main\AD\fig\fig\3.jpg",
    r"D:\MyCourse\PythonFiles\Paper\STAMT-main\AD\fig\fig\4.png"
]

# 自定义保存路径
save_path = r"D:\MyCourse\PythonFiles\Paper\STAMT-main\AD\fig\CLAHE3.png"

# 绘制子图
plot_histograms(image_paths, save_path)
