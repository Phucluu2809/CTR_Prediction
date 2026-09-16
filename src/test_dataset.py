from torch.utils.data import DataLoader
from dataset import CriteoDataset

TRAIN_PATH = "data/processed/train_processed.csv"


def main():
    dataset = CriteoDataset(TRAIN_PATH)
    print(f"Số mẫu trong tập dữ liệu: {len(dataset)}")

    sample = dataset[0]
    print("Các trường của một mẫu:", sample.keys())
    print("dense_x shape:", sample["dense_x"].shape)
    print("sparse_x shape:", sample["sparse_x"].shape)
    print("label:", sample["label"])

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))

    print("\nKích thước của một batch:")
    print("dense_x shape:", batch["dense_x"].shape)
    print("sparse_x shape:", batch["sparse_x"].shape)
    print("label shape:", batch["label"].shape)


if __name__ == "__main__":
    main()
