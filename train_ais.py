import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ShipTrajectoryDataset, pad_collate_fn
from model import TCN_GMP, TCNWithGlobalAttention, TCNWithCrossAttention, TCNWithChannelAttention


# TensorBoard setup
writer = SummaryWriter(log_dir='logs1/TCN_ChannelAttention')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 提取时序数据
def process_files(data_path):
    X_list, Y_list, MMSI_list = [], [], []

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

            # 存储到列表
            X_list.append(X)
            Y_list.append(y)
            MMSI_list.append(mmsi)

    return train_test_split(X_list, Y_list, MMSI_list, test_size=0.2, random_state=2024)


# 类别编码
def encode_labels(Y_train, Y_test):
    label_encoder = LabelEncoder()
    Y_train_enc = label_encoder.fit_transform(Y_train)
    Y_test_enc = label_encoder.transform(Y_test)
    return Y_train_enc, Y_test_enc


def create_data_loaders(X_train, Y_train, X_test, Y_test):
    """Create DataLoader objects for training and testing."""
    train_dataset = ShipTrajectoryDataset(X_train, Y_train)
    test_dataset = ShipTrajectoryDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, class_names=None):
    """Train the model."""
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_Y, batch_mask, original_lengths in train_loader:
            batch_X, batch_Y, batch_mask = batch_X.to(device), batch_Y.to(device), batch_mask.to(device)
            lengths = batch_mask.sum(dim=1).long().cpu()

            optimizer.zero_grad()
            output = model(batch_X, lengths)
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
    y_true, y_pred, misclassified_samples = [], [], []
    test_loss = 0.0

    # 记录开始时间
    start_time = time.time()

    with torch.no_grad():
        for i, (batch_X, batch_Y, batch_mask, original_lengths) in enumerate(test_loader):
            batch_X, batch_Y, batch_mask = batch_X.to(device), batch_Y.to(device), batch_mask.to(device)
            lengths = batch_mask.sum(dim=1).long().cpu()

            output = model(batch_X, lengths)
            loss = criterion(output, batch_Y)  # Compute loss
            test_loss += loss.item()  # Accumulate test loss
            _, predicted = torch.max(output.data, 1)

            y_true.extend(batch_Y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 记录结束时间
    end_time = time.time()

    # 计算每个样本的平均时间
    total_test_time = end_time - start_time
    num_samples = len(test_loader.dataset)
    time_per_sample = total_test_time / num_samples

    print(f'Time per Sample: {time_per_sample:.6f} seconds')

    # Compute metrics
    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')


    # Log metrics to TensorBoard
    writer.add_scalar('Test Loss', avg_test_loss, epoch + 1)  # Log test loss
    writer.add_scalar('Test Accuracy', accuracy, epoch + 1)
    writer.add_scalar('Test Precision', precision, epoch + 1)
    writer.add_scalar('Test Recall', recall, epoch + 1)
    writer.add_scalar('Test F1 Score', f1, epoch + 1)



if __name__ == '__main__':
    # 设置训练和测试集的文件路径
    train_path = 'ais_data/train'

    # 处理训练集和测试集
    X_train, X_test, Y_train, Y_test, MMSI_train, MMSI_test = process_files(train_path)

    # 对标签进行编码
    Y_train_enc, Y_test_enc = encode_labels(Y_train, Y_test)

    # Data loaders
    train_loader, test_loader = create_data_loaders(X_train, Y_train_enc, X_test, Y_test_enc)

    # for batch_idx, (padded_trajectories, labels, attention_mask, original_lengths) in enumerate(train_loader):
    #     print(f"Batch {batch_idx + 1}")
    #     print(f"Padded Trajectories: {padded_trajectories.shape}")
    #     print(f"Labels: {labels}")
    #     print(f"Attention Mask: {attention_mask.shape}")
    #     print(f"Original Lengths: {original_lengths}\n")
    #
    #     # 仅查看第一个批次
    #     break

    # 输出信息
    print(f"X_train shape: {len(X_train)}, X_test shape: {len(X_test)}")
    print(f"Y_train encoded: {Y_train_enc},Y_test encoded: {Y_test_enc} ")
    print(f"MMSI_train: {MMSI_train}, MMSI_test: {MMSI_test}")
    print(X_train[0].shape)

    # Model setup
    input_dim = 4
    num_classes = 3  # 类别数，根据任务

    model = TCNWithChannelAttention(input_dim=input_dim, num_classes=num_classes).to(device)

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