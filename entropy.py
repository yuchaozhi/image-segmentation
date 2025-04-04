import os
import re
import cv2
import numpy as np
import shutil  # 用于复制文件
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from collections import defaultdict

def calculate_image_entropy(image):
    if len(image.shape) > 2:
        image = rgb2gray(image)
        image = img_as_ubyte(image)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def calculate_entropy_for_folder(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    entropy_results = []
    for png_file in png_files:
        file_path = os.path.join(folder_path, png_file)
        image = cv2.imread(file_path)
        if image is None:
            print(f"无法读取文件: {file_path}")
            continue
        entropy = calculate_image_entropy(image)
        entropy_results.append((png_file, entropy))
    return entropy_results

def group_entropy_results(entropy_results):
    groups = defaultdict(list)
    for file_name, entropy in entropy_results:
        match = re.match(r"(OAS1_\d{4}_MR1_\d)\.nii_slice_\d+\.png", file_name)
        if match:
            group_key = match.group(1)
            groups[group_key].append((file_name, entropy))
    return groups

def select_top_entropy_images(grouped_results, source_folder, output_folder, total_images=2496):
    """ 选取熵值最高的图片并保存到指定文件夹 """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 创建目标文件夹
    
    selected_images = []
    total_entropy = {group: sum(ent for _, ent in images) for group, images in grouped_results.items()}
    total_entropy_sum = sum(total_entropy.values())

    # **第一步：按比例选图**
    for group, images in grouped_results.items():
        images.sort(key=lambda x: x[1], reverse=True)
        group_quota = round((total_entropy[group] / total_entropy_sum) * total_images)  # 四舍五入
        selected_images.extend(images[:group_quota])
    
    # **第二步：如果少于 938，补充熵值最高的剩余图片**
    if len(selected_images) < total_images:
        remaining_images = sorted(
            [img for group in grouped_results.values() for img in group if img not in selected_images],
            key=lambda x: x[1], reverse=True
        )
        missing_images = total_images - len(selected_images)
        selected_images.extend(remaining_images[:missing_images])

    # **第三步：如果多于 938，截取前 938 张**
    selected_images = sorted(selected_images, key=lambda x: x[1], reverse=True)[:total_images]

    # **复制文件到目标文件夹**
    for file_name, _ in selected_images:
        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(output_folder, file_name)
        shutil.copy2(source_path, target_path)  # 复制文件

    return selected_images


# 设置文件夹路径
source_folder = r"AD\OASIS\VeryMildDemented\VeryMildDemented"  # 替换为你的源文件夹路径
output_folder = r"AD\OASIS_15000\VeryMildDemented"  # 替换为你的目标文件夹路径

entropy_results = calculate_entropy_for_folder(source_folder)
grouped_results = group_entropy_results(entropy_results)
selected_images = select_top_entropy_images(grouped_results, source_folder, output_folder, total_images=2496)

# 打印最终选择的图片
print(f"\n### 选取的 {len(selected_images)} 张图片已保存到 {output_folder}")
for file_name, entropy in selected_images[:10]:  # 仅打印前10张图片信息
    print(f"{file_name} - 熵: {entropy}")

