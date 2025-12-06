# dataset.py
import os
from typing import List
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T



def list_image_files(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(dirpath, f))
    return files


class ImageFolderDataset(Dataset):
    """
    Simple dataset that loads images from a directory recursively.
    Output tensor in [0,1] (no ImageNet normalization here).
    """
    def __init__(self, root: str, size: int = 256):
        super().__init__()
        self.files = list_image_files(root)
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img


class VimeoPairDataset(Dataset):
    """
    For folders containing exactly 2 frames:
        frame1.png  (t)
        frame2.png  (t+1)

    Subpaths listed in a txt file.
    """
    def __init__(self, root, list_file, resize=256):
        """
        Args:
            root: dataset root directory
            list_file: txt path listing subdirectories
            resize: resize images to square (recommended)
        """
        self.root = root

        with open(list_file, "r") as f:
            self.samples = f.read().splitlines()

        self.transform = T.Compose([
            T.Resize((resize, resize)),
            T.CenterCrop(resize),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sub = self.samples[idx]
        folder = os.path.join(self.root, sub)

        # expect two frames in each folder
        f1 = os.path.join(folder, "img1.png")
        f2 = os.path.join(folder, "img3.png")

        im1 = Image.open(f1).convert("RGB")
        im2 = Image.open(f2).convert("RGB")

        return self.transform(im1), self.transform(im2)

