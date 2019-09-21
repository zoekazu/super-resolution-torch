"""Peak to signal ratio


<explanation>
"""


import abc
import math
from typing import Union

import numpy as np
import torch

from .meta import (ImageEvaluation, ImageEvaluationNdarray,
                   ImageEvaluationTensor)


class PSNR(ImageEvaluation, abc.ABC):

    def calc_psnr(self, src: Union[np.ndarray, torch.Tensor],
                  ref: Union[np.ndarray, torch.Tensor]):
        super()._check_before_calc(src, ref)

        pixel_max, _ = self._get_range(src, ref)
        mse = self._calc_mse(src, ref)
        return 20 * math.log10(float(pixel_max) / math.sqrt(mse))

    @abc.abstractmethod
    def _calc_mse(self, src: Union[np.ndarray, torch.Tensor],
                  ref: Union[np.ndarray, torch.Tensor]) -> float:
        pass


class PSNRTensor(PSNR, ImageEvaluationTensor):
    """peak to signal noise ratio (PSNR) calculator for torch.Tensor
    This module can handle `torch.uint8` or `torch.float`

    """

    def _calc_mse(self, src: torch.Tensor, ref: torch.Tensor,) -> float:
        mse = torch.mean((ref.float() - src.float()) ** 2)
        if mse == 0:
            return 100
        return mse


class PSNRNdarray(PSNR, ImageEvaluationNdarray):

    def _calc_mse(self, src: np.ndarray, ref: np.ndarray) -> float:
        mse = np.mean((ref.astype('float64') - src.astype('float64')) ** 2)
        if mse == 0:
            return 100
        return mse
