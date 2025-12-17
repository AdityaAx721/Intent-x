from datasets import load_dataset
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def download_banking77():
    print("Starting Banking77 download...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("banking77")
    print("Dataset loaded from HuggingFace")

    for split in dataset.keys():
        print(f"Saving {split}.csv")
        df = pd.DataFrame(dataset[split])
        df.to_csv(DATA_DIR / f"{split}.csv", index=False)

    print("Banking77 dataset saved successfully!")

if __name__ == "__main__":
    download_banking77()