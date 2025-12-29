import json
import random
from typing import List

import numpy as np
from PIL import Image


def load_jsonl(path: str) -> List[dict]:
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


class JsonlPairedDataset(object):
    def __init__(
        self,
        entries: List[dict],
        resolution: int,
        resize_scale: float,
        disable_augment: bool,
        disable_resize: bool = False,
    ):
        self.entries = entries
        self.resolution = resolution
        self.resize_scale = resize_scale
        self.disable_augment = disable_augment
        self.disable_resize = disable_resize

    def __len__(self) -> int:
        return len(self.entries)

    def _resample_mode(self):
        if hasattr(Image, "Resampling"):
            return Image.Resampling.BILINEAR
        return Image.BILINEAR

    def _resize_square(self, img: Image.Image, size: int) -> Image.Image:
        return img.resize((size, size), self._resample_mode())

    def _to_tensor(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img).astype(np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))

    def __getitem__(self, idx: int):
        item = self.entries[idx]
        cond_path = item["conditioning_image"]
        gt_path = item["image"]

        cond = Image.open(cond_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        if self.disable_augment:
            if not self.disable_resize:
                cond = self._resize_square(cond, self.resolution)
                gt = self._resize_square(gt, self.resolution)
        else:
            resize_size = int(self.resolution * self.resize_scale)
            cond = self._resize_square(cond, resize_size)
            gt = self._resize_square(gt, resize_size)

            max_offset = max(0, resize_size - self.resolution)
            if max_offset > 0:
                i = random.randint(0, max_offset)
                j = random.randint(0, max_offset)
            else:
                i = 0
                j = 0
            cond = cond.crop((j, i, j + self.resolution, i + self.resolution))
            gt = gt.crop((j, i, j + self.resolution, i + self.resolution))

            if random.random() < 0.5:
                cond = cond.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                cond = cond.transpose(Image.FLIP_TOP_BOTTOM)
                gt = gt.transpose(Image.FLIP_TOP_BOTTOM)

        return self._to_tensor(cond), self._to_tensor(gt)


class FuseDataset(object):
    def __init__(self, datasets: List[JsonlPairedDataset], probabilities: List[float]):
        self.datasets = datasets
        probs = np.array(probabilities, dtype=np.float32)
        probs = probs / probs.sum()
        self.cum_probs = np.cumsum(probs)

    def sample_dataset(self) -> JsonlPairedDataset:
        r = random.random()
        dataset_idx = int(np.searchsorted(self.cum_probs, r))
        return self.datasets[dataset_idx]


class JsonlBatchSampler(object):
    def __init__(self, fuse_dataset: FuseDataset, batch_size: int):
        self.fuse_dataset = fuse_dataset
        self.batch_size = batch_size

    def sample_batch(self, resolution: int):
        xs = np.zeros((self.batch_size, 3, resolution, resolution), dtype=np.float32)
        ys = np.zeros((self.batch_size, 3, resolution, resolution), dtype=np.float32)
        for i in range(self.batch_size):
            dataset = self.fuse_dataset.sample_dataset()
            idx = random.randrange(len(dataset))
            cond, gt = dataset[idx]
            xs[i] = cond
            ys[i] = gt
        return xs, ys


class JsonlConcatDataset(object):
    def __init__(self, datasets: List[JsonlPairedDataset]):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for d in datasets:
            self.offsets.append(total)
            total += len(d)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        for d, offset in zip(self.datasets[::-1], self.offsets[::-1]):
            if idx >= offset:
                return d[idx - offset]
        raise IndexError("index out of range")
