"""Tests for structural similarity warrper for pytorch and numpy


NOTE: SSIM value of scikit-image and matlab are different each other

This implementation is based on this reference
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
Image quality assessment: From error visibility to structural similarity.
IEEE Transactions on Image Processing, 13, 600-612.
https://doi.org/10.1109/TIP.2003.819861
"""

import json
import os

import cv2
import pytest
import torch
from scripts.kpi.ssim import SSIMNdarray, SSIMTensor

LENNA_DIR = os.path.join(os.path.dirname(__file__), '..',
                         'assets', 'lenna')

# NOTE: SSIM of scikit-image and matlab are different each other
ALLOWED_ERROR_FROM_MATLAB = 0.001


def test_ssim_ndarray_float(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMNdarray()
    upsampled_lenna = upsampled_lenna.astype('float64') / 255
    ref_lenna = ref_lenna.astype('float64') / 255
    assert kpi_value_lenna['ssim'] == pytest.approx(
        calculator.calc_ssim(upsampled_lenna, ref_lenna),
        ALLOWED_ERROR_FROM_MATLAB)


def test_ssim_ndarray_uint8(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMNdarray()
    assert kpi_value_lenna['ssim'] == pytest.approx(
        calculator.calc_ssim(upsampled_lenna, ref_lenna),
        ALLOWED_ERROR_FROM_MATLAB)


@pytest.mark.xfail
def test_ssim_ndarray_uint8_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMNdarray()
    added_lenna = upsampled_lenna + 10
    assert kpi_value_lenna['ssim'] == pytest.approx(
        calculator.calc_ssim(added_lenna, ref_lenna),
        ALLOWED_ERROR_FROM_MATLAB)


@pytest.mark.xfail
def test_ssim_ndarray_uint16_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMNdarray()
    changed_lenna = upsampled_lenna.astype('uint16')
    assert kpi_value_lenna['ssim'] == pytest.approx(
        calculator.calc_ssim(changed_lenna, ref_lenna),
        ALLOWED_ERROR_FROM_MATLAB)


def test_ssim_tensor_float(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMTensor()
    upsampled_lenna = torch.from_numpy(upsampled_lenna).float() / 255
    ref_lenna = torch.from_numpy(ref_lenna).float() / 255
    assert kpi_value_lenna['ssim'] == pytest.approx(calculator.calc_ssim(
        upsampled_lenna, ref_lenna),
        ALLOWED_ERROR_FROM_MATLAB)


def test_ssim_tensor_uint8(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMTensor()
    assert kpi_value_lenna['ssim'] == pytest.approx(calculator.calc_ssim(
        torch.from_numpy(upsampled_lenna), torch.from_numpy(ref_lenna)),
        ALLOWED_ERROR_FROM_MATLAB)


@pytest.mark.xfail
def test_ssim_tensor_uint8_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMTensor()
    added_lenna = upsampled_lenna + 10
    assert kpi_value_lenna['ssim'] == pytest.approx(calculator.calc_ssim(
        torch.from_numpy(added_lenna), torch.from_numpy(ref_lenna)),
        ALLOWED_ERROR_FROM_MATLAB)


@pytest.mark.xfail
def test_ssim_tensor_long_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = SSIMTensor()
    changed_lenna = upsampled_lenna.long()
    assert kpi_value_lenna['ssim'] == pytest.approx(calculator.calc_ssim(
        torch.from_numpy(changed_lenna), torch.from_numpy(ref_lenna)),
        ALLOWED_ERROR_FROM_MATLAB)


@pytest.fixture()
def upsampled_lenna():
    return cv2.imread(os.path.join(LENNA_DIR, 'lenna_upsampled.bmp'), cv2.IMREAD_UNCHANGED)


@pytest.fixture()
def ref_lenna():
    return cv2.imread(os.path.join(LENNA_DIR, 'lenna_original.bmp'), cv2.IMREAD_UNCHANGED)


@pytest.fixture()
def kpi_value_lenna():
    with open(os.path.join(LENNA_DIR, 'lenna.json'), 'r') as f:
        return json.load(f)
