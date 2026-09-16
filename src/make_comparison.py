import json
import pandas as pd

LR_RESULT_PATH = "outputs/results/lr_metrics.json"
XGB_RESULT_PATH = "outputs/results/xgb_metrics.json"
DEEPFM_RESULT_PATH = "outputs/results/deepfm_metrics.json"
OUTPUT_PATH = "outputs/results/comparison.csv"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    records = []

    # LR
    lr_result = load_json(LR_RESULT_PATH)
    records.append({
        "model": lr_result.get("model", "LogisticRegression"),
        "valid_auc": lr_result.get("valid_auc"),
        "valid_logloss": lr_result.get("valid_logloss"),
        "test_auc": lr_result.get("test_auc"),
        "test_logloss": lr_result.get("test_logloss")
    })

    # XGBoost
    xgb_result = load_json(XGB_RESULT_PATH)
    records.append({
        "model": xgb_result.get("model", "XGBoost"),
        "valid_auc": xgb_result.get("valid_auc"),
        "valid_logloss": xgb_result.get("valid_logloss"),
        "test_auc": xgb_result.get("test_auc"),
        "test_logloss": xgb_result.get("test_logloss")
    })

    # DeepFM
    deepfm_result = load_json(DEEPFM_RESULT_PATH)
    records.append({
        "model": deepfm_result.get("model", "DeepFM"),
        "valid_auc": deepfm_result.get("best_valid_auc"),
        "valid_logloss": None,
        "test_auc": deepfm_result.get("test_auc"),
        "test_logloss": deepfm_result.get("test_logloss")
    })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Kết quả so sánh đã được lưu tại:")
    print(OUTPUT_PATH)
    print("\nXem trước kết quả:")
    print(df)


if __name__ == "__main__":
    main()
