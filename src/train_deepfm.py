import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss

from dataset import CriteoDataset
from models.deepfm import DeepFM

TRAIN_PATH = "data/processed/train_processed.csv"
VALID_PATH = "data/processed/valid_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"

OUTPUT_MODEL_DIR = "outputs/models"
OUTPUT_RESULT_DIR = "outputs/results"

SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]

BATCH_SIZE = 1024
EPOCHS = 3
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vocab_sizes(train_path):
    df = pd.read_csv(train_path, usecols=SPARSE_FEATURES)
    vocab_sizes = []
    for col in SPARSE_FEATURES:
        vocab_sizes.append(int(df[col].max()) + 1)
    return vocab_sizes


def evaluate(model, dataloader, device):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            dense_x = batch["dense_x"].to(device)
            sparse_x = batch["sparse_x"].to(device)
            labels = batch["label"].to(device)

            logits = model(dense_x, sparse_x)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs)
    logloss = log_loss(all_labels, all_probs)

    return auc, logloss


def main():
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)

    print("Đang tải tập dữ liệu...")
    train_dataset = CriteoDataset(TRAIN_PATH)
    valid_dataset = CriteoDataset(VALID_PATH)
    test_dataset = CriteoDataset(TEST_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Đang thống kê kích thước từ vựng của các đặc trưng sparse...")
    vocab_sizes = get_vocab_sizes(TRAIN_PATH)

    print("Đang khởi tạo mô hình DeepFM...")
    model = DeepFM(vocab_sizes=vocab_sizes, embed_dim=8).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_valid_auc = 0.0
    best_model_path = os.path.join(OUTPUT_MODEL_DIR, "deepfm_best.pth")

    print(f"Bắt đầu huấn luyện, thiết bị = {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            dense_x = batch["dense_x"].to(DEVICE)
            sparse_x = batch["sparse_x"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            logits = model(dense_x, sparse_x)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        valid_auc, valid_logloss = evaluate(model, valid_loader, DEVICE)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Valid AUC: {valid_auc:.6f} "
            f"Valid LogLoss: {valid_logloss:.6f}"
        )

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"Lưu mô hình tốt nhất hiện tại tại: {best_model_path}")

    print("Đang tải mô hình tốt nhất và đánh giá trên tập kiểm thử...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    test_auc, test_logloss = evaluate(model, test_loader, DEVICE)

    print(f"Best Valid AUC: {best_valid_auc:.6f}")
    print(f"Test AUC: {test_auc:.6f}")
    print(f"Test LogLoss: {test_logloss:.6f}")

    result_path = os.path.join(OUTPUT_RESULT_DIR, "deepfm_metrics.json")
    results = {
        "model": "DeepFM",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "best_valid_auc": best_valid_auc,
        "test_auc": test_auc,
        "test_logloss": test_logloss
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Kết quả đánh giá đã được lưu tại: {result_path}")


if __name__ == "__main__":
    main()
