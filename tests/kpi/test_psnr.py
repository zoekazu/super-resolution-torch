"""Tests for peak to signal ratio


<explanation>
"""

import json
import os

import cv2
import pytest
import torch
from scripts.kpi.psnr import PSNRNdarray, PSNRTensor

LENNA_DIR = os.path.join(os.path.dirname(__file__), '..',
                         'assets', 'lenna')


def test_psnr_ndarray_float(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRNdarray()
    upsampled_lenna = upsampled_lenna.astype('float64') / 255
    ref_lenna = ref_lenna.astype('float64') / 255
    assert kpi_value_lenna['psnr'] == pytest.approx(
        calculator.calc_psnr(upsampled_lenna, ref_lenna))


def test_psnr_ndarray_uint8(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRNdarray()
    assert kpi_value_lenna['psnr'] == pytest.approx(
        calculator.calc_psnr(upsampled_lenna, ref_lenna))


@pytest.mark.xfail
def test_psnr_ndarray_uint8_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRNdarray()
    added_lenna = upsampled_lenna + 10
    assert kpi_value_lenna['psnr'] == pytest.approx(
        calculator.calc_psnr(added_lenna, ref_lenna))


@pytest.mark.xfail
def test_psnr_ndarray_uint16_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRNdarray()
    changed_lenna = upsampled_lenna.astype('uint16')
    assert kpi_value_lenna['psnr'] == pytest.approx(
        calculator.calc_psnr(changed_lenna, ref_lenna))


def test_psnr_tensor_float(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRTensor()
    upsampled_lenna = torch.from_numpy(upsampled_lenna).float() / 255
    ref_lenna = torch.from_numpy(ref_lenna).float() / 255
    assert kpi_value_lenna['psnr'] == pytest.approx(calculator.calc_psnr(
        upsampled_lenna, ref_lenna))


def test_psnr_tensor_uint8(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRTensor()
    assert kpi_value_lenna['psnr'] == pytest.approx(calculator.calc_psnr(
        torch.from_numpy(upsampled_lenna), torch.from_numpy(ref_lenna)))


@pytest.mark.xfail
def test_psnr_tensor_uint8_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRTensor()
    added_lenna = upsampled_lenna + 10
    assert kpi_value_lenna['psnr'] == pytest.approx(calculator.calc_psnr(
        torch.from_numpy(added_lenna), torch.from_numpy(ref_lenna)))


@pytest.mark.xfail
def test_psnr_tensor_long_fail(upsampled_lenna, ref_lenna, kpi_value_lenna):
    calculator = PSNRTensor()
    changed_lenna = upsampled_lenna.long()
    assert kpi_value_lenna['psnr'] == pytest.approx(calculator.calc_psnr(
        torch.from_numpy(changed_lenna), torch.from_numpy(ref_lenna)))


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
