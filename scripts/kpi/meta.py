"""Meta file for KPI calculation


This file defines the meta info. for KPI calculation.
"""

import abc
from typing import Union, Tuple

import numpy as np
import torch


class ImageEvaluation(abc.ABC):
    @abc.abstractmethod
    def _check_type(self, src: Union[np.ndarray, torch.Tensor],
                    ref: Union[np.ndarray, torch.Tensor]) -> None:
        raise ValueError('Type of input and reference image must be the same')

    def _check_size(self, src: Union[np.ndarray, torch.Tensor],
                    ref: Union[np.ndarray, torch.Tensor]) -> None:
        if not src.shape == ref.shape:
            raise ValueError('Shape of input and reference image must be the same')

    def _check_channel(self, src: Union[np.ndarray, torch.Tensor],
                       ref: Union[np.ndarray, torch.Tensor]) -> None:
        raise ValueError('image must be 2-dimensional or 1-channel')

    @abc.abstractmethod
    def _get_range(self, src: Union[np.ndarray, torch.Tensor],
                   ref: Union[np.ndarray, torch.Tensor]) -> Tuple[int, int]:
        pass

    def _check_before_calc(self, src, ref) -> None:
        self._check_type(src, ref)
        self._check_size(src, ref)
        self._check_channel(src, ref)


class ImageEvaluationTensor(ImageEvaluation):
    def _check_type(self, src: torch.Tensor, ref: torch.Tensor):
        if src.dtype != ref.dtype:
            super()._check_type(src, ref)

    def _check_channel(self, src: torch.Tensor,
                       ref: torch.Tensor) -> None:
        if src.ndim == 3 and ref.ndim == 3:
            if src.shape[0] == 1 and ref.shape[0] == 1:
                return
        elif src.ndim == 2 and ref.ndim == 2:
            return
        return super()._check_channel(src, ref)

    def _get_range(self, src: torch.Tensor, ref: torch.Tensor) -> Tuple[int, int]:
        if src.dtype == torch.uint8:
            pixel_max = 255
            pixel_min = 0
        elif src.dtype == torch.float:
            if torch.min(src) < 0 or torch.max(src) > 1:
                raise ValueError(
                    'pixel value of input image was higher 1 or lower 0 when dtype is float')
            if torch.min(ref) < 0 or torch.max(ref) > 1:
                raise ValueError(
                    'pixel value of reference image was higher 1 or lower 0 when dtype is float')
            pixel_max = 1
            pixel_min = 0
        else:
            raise ValueError('Type of input and reference image must be either uint8 or float')
        return pixel_max, pixel_min


class ImageEvaluationNdarray(ImageEvaluation):

    def _check_type(self, src: np.ndarray, ref: np.ndarray) -> None:
        if src.dtype != ref.dtype:
            return super()._check_type(src, ref)

    def _get_range(self, src: np.ndarray, ref: np.ndarray) -> Tuple[int, int]:
        if src.dtype == np.uint8:
            pixel_max = 255
            pixel_min = 0
        elif src.dtype == np.float32 or src.dtype == np.float64:
            if np.min(src) < 0 or np.max(src) > 1:
                raise ValueError(
                    'pixel value of input image was higher 1 or lower 0 when dtype is float')
            if np.min(ref) < 0 or np.max(ref) > 1:
                raise ValueError(
                    'pixel value of reference image was higher 1 or lower 0 when dtype is float')
            pixel_max = 1
            pixel_min = 0
        else:
            raise ValueError('Type of input and reference image must be either uint8 or float')
        return pixel_max, pixel_min

    def _check_channel(self, src: np.ndarray,
                       ref: np.ndarray) -> None:
        if src.ndim == 3 and ref.ndim == 3:
            if src.shape[2] == 1 and ref.shape[2] == 1:
                return
        elif src.ndim == 2 and ref.ndim == 2:
            return
        return super()._check_channel(src, ref)
