import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COL = "target"
DENSE_FEATURES = [f"intCol_{i}" for i in range(13)]
SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]


class CriteoDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        # Đặc trưng số: float32
        self.dense_features = torch.tensor(
            self.df[DENSE_FEATURES].values,
            dtype=torch.float32
        )

        # Đặc trưng phân loại: kiểu long, dùng cho embedding ở bước sau
        self.sparse_features = torch.tensor(
            self.df[SPARSE_FEATURES].values,
            dtype=torch.long
        )

        # Nhãn: float32, dùng để tính hàm mất mát phân loại nhị phân
        self.labels = torch.tensor(
            self.df[TARGET_COL].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "dense_x": self.dense_features[idx],
            "sparse_x": self.sparse_features[idx],
            "label": self.labels[idx]
        }
