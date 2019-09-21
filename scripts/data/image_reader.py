"""Image reader for images in directory


This file handles images in a directory.
"""

import glob
import os
from typing import Iterator, List

import cv2
import numpy as np


class ImageReader():

    def __init__(self, dir_path: str, *, file_ext: str = 'bmp'):
        self._dir_path = self._check_end_path(dir_path)
        self._img_paths = self._get_file_paths(dir_path, file_ext)

    def _check_files_exist(self, img_paths: List[str]) -> None:
        if not img_paths:
            raise ValueError(f'There is no images in {self._dir_path}')

    def _get_file_paths(self, dir_path: str, ext: str) -> List[str]:
        return glob.glob(os.path.join(dir_path, "*." + ext))

    def _check_end_path(self, path: str) -> str:
        if path[-1] != os.path.sep:
            return path.rstrip(os.path.sep)
        return path

    def load_file_idx(self, idx: int) -> np.ndarray:
        return cv2.imread(self._img_paths[idx], cv2.IMREAD_UNCHANGED)

    def _load_file(self, img_path: str) -> np.ndarray:
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    def load_files(self) -> Iterator[np.ndarray]:
        for img_path in self._img_paths:
            yield self._load_file(img_path)

    def __len__(self) -> int:
        return len(self._img_paths)

    def get_file_names(self) -> Iterator[str]:
        for img_path in self._img_paths:
            yield img_path


class ImageReaderAsGray(ImageReader):
    def __init__(self, dir_path: str, *, file_ext: str = 'bmp'):
        super().__init__(dir_path, file_ext=file_ext)

    def load_file_idx(self, idx: int) -> np.ndarray:
        return cv2.imread(self._img_paths[idx], cv2.IMREAD_GRAYSCALE)

    def _load_file(self, img_path: str) -> np.ndarray:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


class ImageReaderAsY(ImageReader):
    def __init__(self, dir_path: str, *, file_ext: str = 'bmp'):
        super().__init__(dir_path, file_ext=file_ext)

    def load_file_idx(self, idx: int) -> np.ndarray:
        img = cv2.imread(self._img_paths[idx], cv2.IMREAD_COLOR)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        return img

    def _load_file(self, img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        return img


class ImageReaderAsBool(ImageReader):
    def __init__(self, dir_path: str, *, file_ext: str = 'bmp', threshold_level=127, bool_switch=False):
        super().__init__(dir_path, file_ext=file_ext)
        self._threshould_level = threshold_level
        self._bool_switch = bool_switch

    def load_file_idx(self, idx: int) -> np.ndarray:
        img = cv2.imread(self._img_paths[idx], cv2.IMREAD_GRAYSCALE)

        _, img = cv2.threshold(
            img, self._threshould_level, 255, cv2.THRESH_BINARY)
        bool_img = img.astype(bool)

        if self._bool_switch:
            return np.logical_not(bool_img)
        return bool_img

    def _load_file(self, img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        _, img = cv2.threshold(
            img, self._threshould_level, 255, cv2.THRESH_BINARY)
        bool_img = img.astype(bool)

        if self._bool_switch:
            return np.logical_not(bool_img)
        return bool_img
