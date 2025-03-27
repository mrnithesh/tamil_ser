import pandas as pd
import os

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None, names=["audio_path", "label"])
    # Fix path formatting for Windows/Linux compatibility
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.normpath(x.replace("\\", "/")))
    return df

# Load train/test data
train_df = load_dataset("train_data.csv")
test_df = load_dataset("test_data.csv")

print("Train distribution:\n", train_df["label"].value_counts())
print("\nTest distribution:\n", test_df["label"].value_counts())