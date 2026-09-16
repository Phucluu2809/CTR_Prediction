import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

from metrics import compute_auc, compute_logloss

TRAIN_PATH = "data/processed/train_processed.csv"
VALID_PATH = "data/processed/valid_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"

OUTPUT_MODEL_DIR = "outputs/models"
OUTPUT_RESULT_DIR = "outputs/results"

TARGET_COL = "target"
FEATURE_COLS = [f"intCol_{i}" for i in range(13)] + [f"catCol_{i}" for i in range(26)]


def main():
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)

    print("Đang đọc dữ liệu huấn luyện/xác thực/kiểm thử...")
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_valid = valid_df[FEATURE_COLS]
    y_valid = valid_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print("Bắt đầu huấn luyện Logistic Regression...")
    model = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    print("Đang đánh giá trên tập xác thực...")
    valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_auc = compute_auc(y_valid, valid_pred)
    valid_logloss = compute_logloss(y_valid, valid_pred)

    print("Đang đánh giá trên tập kiểm thử...")
    test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = compute_auc(y_test, test_pred)
    test_logloss = compute_logloss(y_test, test_pred)

    print(f"Valid AUC: {valid_auc:.6f}")
    print(f"Valid LogLoss: {valid_logloss:.6f}")
    print(f"Test AUC: {test_auc:.6f}")
    print(f"Test LogLoss: {test_logloss:.6f}")

    model_path = os.path.join(OUTPUT_MODEL_DIR, "lr_model.joblib")
    result_path = os.path.join(OUTPUT_RESULT_DIR, "lr_metrics.json")

    dump(model, model_path)

    results = {
        "model": "LogisticRegression",
        "valid_auc": valid_auc,
        "valid_logloss": valid_logloss,
        "test_auc": test_auc,
        "test_logloss": test_logloss
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Mô hình đã được lưu tại:", model_path)
    print("Kết quả đánh giá đã được lưu tại:", result_path)


if __name__ == "__main__":
    main()
