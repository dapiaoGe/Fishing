import os
import time
from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ShipTrajectoryImageDataset, pad_collate_fn_add_image
from model import TCN_GMP, TCNWithGlobalAttention, TCNWithCrossAttention, TCNWithChannelAttention, TCNWithGlobalAttention_EfficientNet, TCNWithGlobalAttention_EfficientNet_Select
from PIL import Image
import torch
from torchvision import transforms

# TensorBoard setup
writer = SummaryWriter(log_dir='logs2/TCN_GA_EfficientNet2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 载入图像
def load_image(image_path, image_size=(224, 224)):
    """加载图像并调整大小，返回张量格式。"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    return transform(image)

# 提取时序数据
def process_files(data_path,image_path):
    X_list, Y_list, MMSI_list, all_image = [], [], [], []

    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)

            # 提取 4 维特征：lat, lon, 速度, 方向
            X = df[['lat', 'lon', '速度', '方向']].values

            # 提取 type 作为标签 (假设所有行 type 相同)
            y = df['type'].iloc[0]

            # 文件名作为船舶ID（MMSI）
            mmsi = os.path.splitext(file)[0]

            # 加载生成的图像
            image_file = os.path.join(image_path, f'{mmsi}.png')
            if os.path.exists(image_file):
                image_tensor = load_image(image_file)
                all_image.append(image_tensor)

            # 存储到列表
            X_list.append(X)
            Y_list.append(y)
            MMSI_list.append(mmsi)

    return train_test_split(X_list, Y_list, MMSI_list, all_image, test_size=0.2, random_state=2024)


# 类别编码
def encode_labels(Y_train, Y_test):
    label_encoder = LabelEncoder()
    Y_train_enc = label_encoder.fit_transform(Y_train)
    Y_test_enc = label_encoder.transform(Y_test)
    return Y_train_enc, Y_test_enc


def create_data_loaders(X_train, Y_train, X_test, Y_test, image_train, image_test):
    """Create DataLoader objects for training and testing."""
    train_dataset = ShipTrajectoryImageDataset(X_train, image_train, Y_train)
    test_dataset = ShipTrajectoryImageDataset(X_test, image_test,Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=pad_collate_fn_add_image)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=pad_collate_fn_add_image)

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, class_names=None):
    """Train the model."""
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_images, batch_Y, batch_mask, original_lengths in train_loader:
            batch_X, batch_images, batch_Y, batch_mask = batch_X.to(device), batch_images.to(device), batch_Y.to(device), batch_mask.to(device)
            lengths = batch_mask.sum(dim=1).long().cpu()

            optimizer.zero_grad()
            output = model(batch_X, lengths, batch_images, use_images=True, use_ais=True)
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        # Log the average loss for this epoch to TensorBoard
        writer.add_scalar('Training Loss', avg_loss, epoch + 1)

        # Evaluate model after each epoch and log metrics
        if class_names:
            evaluate_model(model, test_loader, class_names, epoch)

def evaluate_model(model, test_loader, class_names, epoch):
    """Evaluate the model and print metrics."""
    model.eval()
    y_true, y_pred = [], []
    y_score = []
    test_loss = 0.0

    start_time = time.time()

    with torch.no_grad():
        for i, (batch_X, batch_images, batch_Y, batch_mask, original_lengths) in enumerate(test_loader):
            batch_X, batch_images, batch_Y, batch_mask = batch_X.to(device), batch_images.to(device), batch_Y.to(
                device), batch_mask.to(device)
            lengths = batch_mask.sum(dim=1).long().cpu()

            output = model(batch_X, lengths, batch_images, use_images=True, use_ais=True)
            loss = criterion(output, batch_Y)  # Compute loss
            test_loss += loss.item()  # Accumulate test loss

            _, predicted = torch.max(output.data, 1)
            y_true.extend(batch_Y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.append(output.softmax(dim=1).cpu().numpy())  # Store predicted probabilities

    end_time = time.time()
    total_test_time = end_time - start_time
    num_samples = len(test_loader.dataset)
    time_per_sample = total_test_time / num_samples

    print(f'Time per Sample: {time_per_sample:.6f} seconds')

    # Convert y_score from list to array
    y_score = np.concatenate(y_score)  # Concatenate the list of arrays

    # Compute metrics
    avg_test_loss = test_loss / len(test_loader)

    # ROC and AUC calculation
    y_true_one_hot = np.eye(len(class_names))[y_true]
    plt.figure(figsize=(10, 8))

    # Compute macro-average and micro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([roc_curve(y_true_one_hot[:, i], y_score[:, i])[0] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)

    # Interpolating per-class ROC
    for i, color in zip(range(len(class_names)), cycle(['blue', 'green', 'red', 'purple', 'orange'])):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        mean_tpr += np.interp(all_fpr, fpr, tpr)  # Using np.interp

    # Finalizing macro-average ROC
    mean_tpr /= len(class_names)
    macro_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, 'k--', label=f'Macro-average (AUC = {macro_auc:.2f})', linewidth=2)

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, 'k:', label=f'Micro-average (AUC = {roc_auc_micro:.2f})', linewidth=2)

    # ROC curve styling
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()

    # Save ROC curve
    plt.savefig(f'logs2/ROC_AUC_epoch_{epoch + 1}.png', dpi=300)
    plt.close()

    # Logging to TensorBoard
    writer.add_scalar('Test Loss', avg_test_loss, epoch + 1)
    writer.add_scalar('Macro-average AUC', macro_auc, epoch + 1)
    writer.add_scalar('Micro-average AUC', roc_auc_micro, epoch + 1)


if __name__ == '__main__':
    # 设置训练和测试集的文件路径
    train_path = 'ais_data/train'
    image_path = 'generated_images'

    # 处理训练集和测试集
    X_train, X_test, Y_train, Y_test, MMSI_train, MMSI_test, image_train, image_test = process_files(train_path,image_path)

    # 对标签进行编码
    Y_train_enc, Y_test_enc = encode_labels(Y_train, Y_test)

    # Data loaders
    train_loader, test_loader = create_data_loaders(X_train, Y_train_enc, X_test, Y_test_enc, image_train, image_test)

    # for batch_idx, (padded_trajectories, batch_images, labels, attention_mask, original_lengths) in enumerate(train_loader):
    #     print(f"Batch {batch_idx + 1}")
    #     print(f"Padded Trajectories: {padded_trajectories.shape}")
    #     print(f"Images:{batch_images.shape}")
    #     print(f"Labels: {labels}")
    #     print(f"Attention Mask: {attention_mask.shape}")
    #     print(f"Original Lengths: {original_lengths}\n")
    #
    #     # 仅查看第一个批次
    #     break
    #
    # # 输出信息
    # print(f"X_train shape: {len(X_train)}, X_test shape: {len(X_test)}")
    # print(f"Y_train encoded: {Y_train_enc},Y_test encoded: {Y_test_enc} ")
    # print(f"MMSI_train: {MMSI_train}, MMSI_test: {MMSI_test}")
    # print(X_train[0].shape)

    # Model setup
    input_dim = 4
    num_classes = 3  # 类别数

    model = TCNWithGlobalAttention_EfficientNet_Select(input_dim=input_dim, num_classes=num_classes).to(device)

    # Loss function with class weights
    class_counts = np.unique(Y_train_enc, return_counts=True)[1]
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    class_names = ['Ciwang', 'Weiwang', 'TuoWang']
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, class_names=class_names)

    # Close the TensorBoard writer
    writer.close()