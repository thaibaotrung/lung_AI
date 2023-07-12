"""
OASIS dataset processed at https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md
"""

import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple
import zipfile
import numpy as np
import nibabel as nib
import PIL
import torch
from torch.utils.data import Dataset
from typing import Literal, Optional, Tuple


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = (nib.load(path).get_fdata() * 255).astype(np.uint8).squeeze()
    img = PIL.Image.fromarray(img)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)/255
    img = np.rot90(img, -1)
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = nib.load(path).get_fdata().astype(np.int8).squeeze()
    seg = PIL.Image.fromarray(seg)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = seg.astype(np.float32)
    seg = np.rot90(seg, -1)
    return seg.copy()


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
        zip_file = dest_folder / 'subset1.zip'
        
        subprocess.run(
            ["curl", zip_url, "--create-dirs", "-o", str(zip_file)],
            stderr=subprocess.DEVNULL,
            check=True,
        )

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)

    return dest_folder


@dataclass
class LUNADataset(Dataset):
    split: Literal["support", "test"]
    label: Optional[Literal["Nodule", "Bengin", "background"]] = None
    support_frac: float = 0.7

    def __post_init__(self):
        path = require_download_luna()
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(path)]
        if self.label is not None:
            self._ilabel = {"Nodule": 1, "Bengin": 2, "background": 0}[self.label]
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
            seg = seg[self._ilabel][None]
        return img, seg
