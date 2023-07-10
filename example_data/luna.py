import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = np.stack([seg == 0, seg == 128, seg == 255])
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("*.mhd")):
        img = process_img(file, size=size)
        seg_file = file.with_suffix(".raw")
        seg = process_seg(seg_file, size=size)
        data.append((img / 255.0, seg))
    return data


def require_download_luna():
    dest_folder = pathlib.Path("/tmp/universeg_luna/")

    if not dest_folder.exists():
        zip_url = "https://zenodo.org/record/3723295/files/subset1.zip?download=1"
        subprocess.run(
            ["curl", zip_url, "--create-dirs", "-o",
                str(dest_folder/'subset1.zip'),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            ["zip", 'xf', str(
                dest_folder/'subset.zip'), '-C', str(dest_folder)],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder



@dataclass
class LUNADataset(Dataset):
    split: Literal["train", "test"]
    label: Optional[Literal["Malignant", "Nodule", "Benign"]] = None
    support_frac: float = 0.7

    def __post_init__(self):
        path = require_download_luna()
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(path)]
        if self.label is not None:
            self._ilabel = {"Malignant": 1, "Nodule": 2, "Benign": 0}[self.label]
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"train": p[:i], "test": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = seg[self._ilabel][None]
        return img, seg
