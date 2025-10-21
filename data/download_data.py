import os
from torch import ErrorReport
import yaml
import shutil
import tqdm
import urllib.request
import kagglehub

from pathlib import Path


def load_yaml():
    yaml_path = Path("data/multiplane.yaml")
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        train_dir = cfg["train"]
        val_dir = cfg["val"]
        link = cfg["download"]

        return (train_dir, val_dir), link


def ensure_dir(dir: str):
    if not os.path.exists(Path(dir)):
        print(f"❌ No such directory found: {dir}")
        os.makedirs(dir)
        print(f"└── ✅ Directory made at: {dir}")

    return Path(dir)


def download_dataset(dirs: tuple[str, str], dataset_name: str):
    dataset_dir = f"data/{dataset_name}"
    ensure_dir(dataset_dir)

    if dataset_name == "RSNA-MICCAI":
        handle = "davidbroberts/brain-tumor-object-detection-datasets"
    elif dataset_name == "Br35H":
        handle = "ahmedhamada0/brain-tumor-detection"
    else:
        raise KeyError(f"Unavailable dataset name: {dataset_name}\n")

    path = kagglehub.dataset_download(handle)
    print(f"📀 {dataset_name} downloaded\n└── ✅ Saved at: {path}")

    dest_root = Path(f"data/{dataset_name}")
    path = Path(path)
    subdirs = [dir for dir in path.iterdir() if dir.is_dir()]
    print(f"📁 Copying downloaded files to {dest_root}...")
    for i, dir in enumerate(subdirs, 1):
        dest_path = dest_root / dir.name
        shutil.copytree(src=dir, dst=dest_path, dirs_exist_ok=True)
        if i == len(subdirs):
            print(f"└── ✅ {dir.name} copied")
        else:
            print(f"├── ✅ {dir.name} copied")

    return dirs


def download_pretrained_weight(link: str):
    file_name = "V9back_1kpretrained_timm_style.pth"
    dest_root = ensure_dir("data/pretrain")
    dest_path = dest_root / file_name

    print(f"📁 Downloading {file_name} at {dest_root}...")
    try:
        urllib.request.urlretrieve(link, dest_path)
        print(f"└── ✅ Weights downloaded at {dest_path}")
    except Exception:
        raise Exception("└── ❌ Failed to download weights")


if __name__ == "__main__":
    dirs, link = load_yaml()
    download_dataset(dirs, "RSNA-MICCAI")
    download_dataset(dirs, "Br35H")
    download_pretrained_weight(link)
