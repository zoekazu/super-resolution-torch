"""Dataset loader


This file defines the dataset loader for pytorch
"""

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .image_reader import ImageReader
from .utils import modcrop, shave


@dataclass
class TrainDatasetParam():
    stride_size: int
    shave_size: int
    scale: int
    input_size: int = None
    label_size: int = None

    def __post_init__(self):
        if not bool(self.label_size) ^ bool(self.input_size):
            raise ValueError

        if self.input_size:
            self.label_size = self.input_size * self.scale - self.shave_size * self.scale * 2

        elif self.label_size:
            if self.label_size % self.scale != 0:
                raise ValueError
            self.input_size = self.label_size // self.scale + self.shave_size * 2


@dataclass(frozen=True)
class Position():
    ymin: int
    ymax: int
    xmin: int
    xmax: int


class ValidateDataset(Dataset):
    def __init__(self, reader: ImageReader, scale: int, shave_size: int, transform=None):
        self._reader = reader
        self._scale = scale
        self._shave_size = shave_size
        self._transform = transform

    def __getitem__(self, index):
        hr_img = self._reader.load_file_idx(index)

        cropped_hr_img = modcrop(hr_img, self._scale)

        cnn_label = shave(cropped_hr_img,
                          self._shave_size * self._scale,
                          self._shave_size * self._scale)

        cnn_input = cv2.resize(cropped_hr_img, dsize=None,
                               fx=1 / self._scale,
                               fy=1 / self._scale,
                               interpolation=cv2.INTER_CUBIC)

        # cnn_input = Image.fromarray(cnn_input)
        # cnn_label = Image.fromarray(cnn_label)
        cnn_input = cnn_input.astype(np.float32) / 255
        cnn_label = cnn_label.astype(np.float32) / 255

        if self._transform:
            return self._transform(cnn_input), self._transform(cnn_label)
        return cnn_input, cnn_label

    def __len__(self):
        return len(self._reader)


class TrainDataset(Dataset):
    def __init__(self, reader: ImageReader, loader_param: TrainDatasetParam,
                 transform=None):
        self._transform = transform
        self._reader = reader
        self._loader_param = loader_param
        self._idx_position = self._get_idx_position(loader_param)

    def _get_idx_position(self, loader_param: TrainDatasetParam):
        cnt = 0
        idx_position = {}
        for img_num, hr_img in enumerate(self._reader.load_files()):
            cropped_hr_img = modcrop(hr_img, loader_param.scale)
            hei, wid = cropped_hr_img.shape

            for y, x in itertools.product(range(0,
                                                hei - loader_param.input_size * loader_param.scale,
                                                loader_param.stride_size),
                                          range(0,
                                                wid - loader_param.input_size * loader_param.scale,
                                                loader_param.stride_size)):
                idx_position[cnt] = {
                    "img_num": img_num,
                    "position":
                        Position(ymin=y,
                                 ymax=y + loader_param.input_size * loader_param.scale,
                                 xmin=x,
                                 xmax=x + loader_param.input_size * loader_param.scale
                                 )
                }
                cnt += 1
        return idx_position

    def _get_pair(self, index):
        img_num = self._idx_position[index]["img_num"]
        position = self._idx_position[index]["position"]

        hr_img = self._reader.load_file_idx(img_num)

        hr_patch = hr_img[position.ymin: position.ymax,
                          position.xmin: position.xmax].copy()

        hr_patch_shaved = shave(hr_patch,
                                self._loader_param.shave_size * self._loader_param.scale,
                                self._loader_param.shave_size * self._loader_param.scale)

        lr_patch = cv2.resize(hr_patch, dsize=None,
                              fx=1 / self._loader_param.scale,
                              fy=1 / self._loader_param.scale,
                              interpolation=cv2.INTER_CUBIC)

        return lr_patch, hr_patch_shaved

    def __getitem__(self, idx):
        cnn_input, cnn_label = self._get_pair(idx)

        cnn_input = cnn_input.astype(np.float32) / 255
        cnn_label = cnn_label.astype(np.float32) / 255

        # cnn_input = Image.fromarray(cnn_input)
        # cnn_label = Image.fromarray(cnn_label)

        if self._transform:
            return self._transform(cnn_input), self._transform(cnn_label)
        return cnn_input, cnn_label

    def __len__(self):
        return len(self._idx_position)


def recursive_len_list(list_):
    cnt = 0
    if isinstance(list_, list):
        for v in list_:
            cnt += recursive_len_list(v)
        return cnt
    else:
        return 1
