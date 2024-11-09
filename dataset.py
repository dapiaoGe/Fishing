import torch
from torch.utils.data import Dataset

# AIS数据加载器类
class ShipTrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        trajectory = self.X[idx]
        label = self.Y[idx]
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return trajectory, label

def pad_collate_fn(batch):
    trajectories, labels = zip(*batch)  # 解包批次数据

    # 获取当前批次中轨迹的最大长度
    max_len = max([traj.shape[0] for traj in trajectories])  # 使用 shape 而不是 size

    padded_trajectories = torch.zeros((len(trajectories), max_len, trajectories[0].shape[1]))
    attention_mask = torch.zeros((len(trajectories), max_len))
    original_lengths = []

    for i, traj in enumerate(trajectories):
        end = len(traj)
        padded_trajectories[i, :end, :] = torch.tensor(traj)
        attention_mask[i, :end] = 1
        original_lengths.append(len(traj))

    labels = torch.tensor(labels, dtype=torch.long)
    original_lengths = torch.tensor(original_lengths, dtype=torch.long)

    return padded_trajectories, labels, attention_mask, original_lengths




# AIS+Image数据加载器类
class ShipTrajectoryImageDataset(Dataset):
    def __init__(self, X_trajectory, X_image, Y):
        self.X_trajectory = X_trajectory
        #print(f"Original image shape: {X_image[0].shape}")  # 输出原始图像的 shape
        self.X_image = X_image
        self.Y = Y

    def __len__(self):
        return len(self.X_trajectory)

    def __getitem__(self, idx):
        trajectory = self.X_trajectory[idx]
        image = self.X_image[idx]
        label = self.Y[idx]
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32) # (C, H, W) 格式
        label = torch.tensor(label, dtype=torch.long)
        return trajectory, image, label

def pad_collate_fn_add_image(batch):
    trajectories, images, labels = zip(*batch)  # 解包批次数据

    # 获取当前批次中轨迹的最大长度
    max_len = max([traj.shape[0] for traj in trajectories])  # 使用 shape 而不是 size

    padded_trajectories = torch.zeros((len(trajectories), max_len, trajectories[0].shape[1]))
    attention_mask = torch.zeros((len(trajectories), max_len))
    original_lengths = []

    for i, traj in enumerate(trajectories):
        end = len(traj)
        padded_trajectories[i, :end, :] = torch.tensor(traj)
        attention_mask[i, :end] = 1
        original_lengths.append(len(traj))

    labels = torch.tensor(labels, dtype=torch.long)
    original_lengths = torch.tensor(original_lengths, dtype=torch.long)

    images = torch.stack(images)  # 图像数据堆叠

    return padded_trajectories, images, labels, attention_mask, original_lengths