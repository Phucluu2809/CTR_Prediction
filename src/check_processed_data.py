import pandas as pd

TARGET_COL = "target"
DENSE_FEATURES = [f"intCol_{i}" for i in range(13)]
SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]

TRAIN_PATH = "data/processed/train_processed.csv"
VALID_PATH = "data/processed/valid_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"


def check_file(path, name):
    print(f"\n========== Kiểm tra {name} ==========")
    df = pd.read_csv(path)

    print(f"Kích thước {name}: {df.shape}")
    print(f"3 dòng đầu của {name}:")
    print(df.head(3))

    print(f"\nTổng số giá trị thiếu của {name}: {df.isnull().sum().sum()}")
    print(f"Phân bố target của {name}:")
    print(df[TARGET_COL].value_counts(normalize=True))

    print(f"\nThống kê mẫu các đặc trưng số của {name}:")
    print(df[DENSE_FEATURES].describe().iloc[:2])  # Chỉ xem count / mean

    print(f"\nVí dụ về đặc trưng phân loại của {name}:")
    print(df[SPARSE_FEATURES[:3]].head(3))


def main():
    check_file(TRAIN_PATH, "train")
    check_file(VALID_PATH, "valid")
    check_file(TEST_PATH, "test")


if __name__ == "__main__":
    main()
