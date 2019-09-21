"""Utilities for image processing


The file includes image processing functions
"""


import numpy as np


def modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    if img.ndim == 3:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1], :]
    elif img.ndim == 2:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1]]
    else:
        raise ValueError
    return out_img


def shave(img: np.ndarray, shave_y: int, shave_x: int):
    if img.ndim == 3:
        return img[shave_y: -shave_y,
                   shave_x: -shave_x, :]
    elif img.ndim == 2:
        return img[shave_y: -shave_y,
                   shave_x: -shave_x]
    else:
        raise ValueError
