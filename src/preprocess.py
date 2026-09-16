import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# 1. Cấu hình cơ bản
# =========================
RAW_DATA_PATH = "data/raw/Criteo_1M_with_nans.csv"
PROCESSED_DIR = "data/processed"

TARGET_COL = "target"
DENSE_FEATURES = [f"intCol_{i}" for i in range(13)]
SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]


def main():
    # Tạo thư mục đầu ra
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Bắt đầu đọc dữ liệu gốc...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Kích thước dữ liệu gốc: {df.shape}")

    # =========================
    # 2. Xử lý đặc trưng số
    # =========================
    print("Đang xử lý đặc trưng số...")
    df[DENSE_FEATURES] = df[DENSE_FEATURES].fillna(0)

    scaler = StandardScaler()
    df[DENSE_FEATURES] = scaler.fit_transform(df[DENSE_FEATURES])

    # =========================
    # 3. Xử lý đặc trưng phân loại
    # =========================
    print("Đang xử lý đặc trưng phân loại...")
    df[SPARSE_FEATURES] = df[SPARSE_FEATURES].fillna("missing").astype(str)

    for col in SPARSE_FEATURES:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # =========================
    # 4. Xử lý cột nhãn
    # =========================
    print("Đang xử lý cột nhãn...")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # =========================
    # 5. Chia tập dữ liệu
    # train : valid : test = 8 : 1 : 1
    # =========================
    print("Đang chia tập huấn luyện/xác thực/kiểm thử...")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[TARGET_COL]
    )

    print(f"Kích thước tập huấn luyện: {train_df.shape}")
    print(f"Kích thước tập xác thực: {valid_df.shape}")
    print(f"Kích thước tập kiểm thử: {test_df.shape}")

    # =========================
    # 6. Lưu kết quả
    # =========================
    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    valid_path = os.path.join(PROCESSED_DIR, "valid_processed.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_processed.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Tiền xử lý hoàn tất, các tệp đã được lưu:")
    print(train_path)
    print(valid_path)
    print(test_path)


if __name__ == "__main__":
    main()
