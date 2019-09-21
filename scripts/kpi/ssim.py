"""Structural similarity warrper for pytorch and numpy


NOTE: SSIM value of scikit-image and matlab are different each other

This implementation is based on this reference
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
Image quality assessment: From error visibility to structural similarity.
IEEE Transactions on Image Processing, 13, 600-612.
https://doi.org/10.1109/TIP.2003.819861
"""

import abc
from typing import Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from .meta import ImageEvaluationNdarray, ImageEvaluationTensor


class SSIM():
    """
    Notes:
        To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
        to True, `sigma` to 1.5, and `use_sample_covariance` to False.
        .. versionchanged:: 0.16
            This function was renamed from ``skimage.measure.compare_ssim`` to
            ``skimage.metrics.structural_similarity``.

    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
        (2004). Image quality assessment: From error visibility to
        structural similarity. IEEE Transactions on Image Processing,
        13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        :DOI:`10.1109/TIP.2003.819861`
    """

    @abc.abstractmethod
    def calc_ssim(self, src: Union[np.ndarray, torch.Tensor],
                  ref: Union[np.ndarray, torch.Tensor]) -> float:
        pass


class SSIMTensor(SSIM, ImageEvaluationTensor):
    """peak to signal noise ratio (PSNR) calculator for torch.Tensor
    This module can handle `torch.uint8` or `torch.float`

    """

    def calc_ssim(self, src: torch.Tensor, ref: torch.Tensor) -> float:
        super()._check_before_calc(src, ref)

        pixel_max, pixel_min = self._get_range(src, ref)
        src_array = src.to('cpu').detach().numpy().copy()
        ref_array = ref.to('cpu').detach().numpy().copy()
        return ssim(src_array, ref_array, data_range=float(pixel_max - pixel_min),
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


class SSIMNdarray(SSIM, ImageEvaluationNdarray):

    def calc_ssim(self, src: np.ndarray, ref: np.ndarray) -> float:
        super()._check_before_calc(src, ref)

        pixel_max, pixel_min = self._get_range(src, ref)
        return ssim(src, ref, data_range=float(pixel_max - pixel_min),
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
