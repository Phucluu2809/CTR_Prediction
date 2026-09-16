from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .models.deepfm import DeepFM
except ImportError:
    from models.deepfm import DeepFM


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "outputs" / "models" / "deepfm_best.pth"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "test_processed.csv"

DENSE_FEATURES = [f"intCol_{i}" for i in range(13)]
SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]


class CTRPredictor:
    def __init__(self, model_path=MODEL_PATH, device="cpu"):
        self.device = torch.device(device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        vocab_sizes = [
            state_dict[f"fm_embeddings.{i}.weight"].shape[0]
            for i in range(len(SPARSE_FEATURES))
        ]
        embed_dim = state_dict["fm_embeddings.0.weight"].shape[1]
        hidden_units = self._hidden_units_from_state_dict(state_dict)

        self.model = DeepFM(
            vocab_sizes=vocab_sizes,
            embed_dim=embed_dim,
            hidden_units=hidden_units
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.vocab_sizes = vocab_sizes

    @staticmethod
    def _hidden_units_from_state_dict(state_dict):
        linear_layers = []
        for key, value in state_dict.items():
            if key.startswith("dnn.") and key.endswith(".weight"):
                layer_index = int(key.split(".")[1])
                linear_layers.append((layer_index, value.shape[0]))
        return [size for _, size in sorted(linear_layers)]

    def predict(self, frame):
        dense_x = torch.tensor(
            frame[DENSE_FEATURES].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device
        )
        sparse_values = frame[SPARSE_FEATURES].to_numpy(dtype=np.int64)
        for index, vocab_size in enumerate(self.vocab_sizes):
            sparse_values[:, index] = np.clip(sparse_values[:, index], 0, vocab_size - 1)
        sparse_x = torch.tensor(sparse_values, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.predict_components(dense_x, sparse_x)

        return {
            name: values.detach().cpu().numpy()
            for name, values in outputs.items()
        }


def load_demo_pool(limit=3000):
    return pd.read_csv(TEST_DATA_PATH, nrows=limit)
