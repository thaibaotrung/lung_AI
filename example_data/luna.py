import csv
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Literal

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32) / 255
    img = np.rot90(img, -1)
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = seg.astype(np.float32)
    seg = np.rot90(seg, -1)
    return seg.copy()


def load_folder(csv_file: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            img_path = pathlib.Path(row[0])
            seg_path = pathlib.Path(row[1])
            img = process_img(img_path, size=size)
            seg = process_seg(seg_path, size=size)
            data.append((img, seg))
    return data


@dataclass
class LUNADataset(Dataset):
    split: Literal["support", "test"]
    label: int
    support_frac: float = 0.7

    def __post_init__(self):
        csv_file = pathlib.Path("https://zenodo.org/record/3723295/files/annotations.csv?download=1")  # Thay đổi đường dẫn đến tệp .csv của bạn tại đây
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(csv_file)]
        if self.label is not None:
            self._ilabel = self.label
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = (seg == self._ilabel)[None]
        return img, seg
