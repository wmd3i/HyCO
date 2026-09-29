from huggingface_hub import snapshot_download

# Replace local_dir with the directory where the dataset should be stored
local_dir = '../data/ml4co_mis'

snapshot_download(
    repo_id="ML4CO/ML4CO-Bench-101-SL",
    repo_type="dataset",
    allow_patterns=["train_dataset/mis/*"],
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

# print(f"Dataset files downloaded to: {local_dir}")

snapshot_download(
    repo_id="ML4CO/ML4CO-Bench-101-SL",
    repo_type="dataset",
    allow_patterns=["val_dataset/mis/*"],
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"Dataset files downloaded to: {local_dir}")


snapshot_download(
    repo_id="ML4CO/ML4CO-Bench-101-SL",
    repo_type="dataset",
    allow_patterns=["test_dataset/mis/*"],
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"Dataset files downloaded to: {local_dir}")