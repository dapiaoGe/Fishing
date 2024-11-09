import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_trajectory_image(lat, lon, mmsi, save_path):
    """根据lat和lon生成轨迹图像并保存。"""
    plt.figure()
    plt.plot(lon, lat, marker='o', markersize=2, linestyle='-', color='b')

    # 隐藏坐标轴的刻度
    plt.xticks([])  # 隐藏X轴刻度
    plt.yticks([])  # 隐藏Y轴刻度

    # 移除坐标轴框架
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存图像
    image_file = os.path.join(save_path, f'{mmsi}.png')
    plt.savefig(image_file, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_and_save_images(data_path, save_image_path):
    """处理CSV文件并生成轨迹图像。"""
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)

            # 提取经纬度数据
            lat = df['lat'].values
            lon = df['lon'].values

            # 文件名作为船舶ID（MMSI）
            mmsi = os.path.splitext(file)[0]

            # 生成并保存轨迹图像
            generate_trajectory_image(lat, lon, mmsi, save_image_path)

# 生成并保存所有轨迹图像
data_path = 'ais_data/train'  # AIS数据路径
save_image_path = 'generated_images'  # 保存图像的路径
process_and_save_images(data_path, save_image_path)
