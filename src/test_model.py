import torch
from models.deepfm import DeepFM
from dataset import CriteoDataset

TRAIN_PATH = "data/processed/train_processed.csv"


def get_vocab_sizes(df):
    sparse_features = [f"catCol_{i}" for i in range(26)]
    vocab_sizes = []
    for col in sparse_features:
        vocab_sizes.append(int(df[col].max()) + 1)
    return vocab_sizes


def main():
    dataset = CriteoDataset(TRAIN_PATH)
    df = dataset.df
    vocab_sizes = get_vocab_sizes(df)

    model = DeepFM(vocab_sizes=vocab_sizes, embed_dim=8)

    sample = dataset[0]
    dense_x = sample["dense_x"].unsqueeze(0)   # [1, 13]
    sparse_x = sample["sparse_x"].unsqueeze(0) # [1, 26]

    output = model(dense_x, sparse_x)

    print("Kích thước đầu ra của mô hình:", output.shape)
    print("Giá trị đầu ra của mô hình:", output)


if __name__ == "__main__":
    main()
