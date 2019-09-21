"""Tests for total valiation


<explanation>
"""

import pytest
import torch
from scripts.loss.total_variation import (TotalVariation, TotalVariationL1Loss,
                                          TotalVariationMSELoss)


@pytest.fixture()
def sample_tensor():
    list_ = [float(i) / 100 for i in range(100)] * 4
    return torch.Tensor(list_).reshape(4, 1, 10, 10)


@pytest.fixture()
def sample_tensor_x2():
    list_ = [float(i * 2) / 100 for i in range(100)] * 4
    return torch.Tensor(list_).reshape(4, 1, 10, 10)


class TestTotalVariation():

    def test_forward(self, sample_tensor):
        calculator = TotalVariation(is_mean_reduction=True)
        tv = calculator(sample_tensor)
        assert tv.shape == torch.Size([sample_tensor.shape[0]])
        assert torch.equal(tv, torch.Tensor([0.1100, 0.1100, 0.1100, 0.1100]))


class TestTotalVariationMSELoss():

    def test_forward(self, sample_tensor, sample_tensor_x2):
        calculator = TotalVariationMSELoss(is_mean_reduction=True)
        tv_loss = calculator(sample_tensor, sample_tensor_x2)
        assert tv_loss.item() == 0.01209999993443489


class TestTotalVariationL1Loss():

    def test_forward(self, sample_tensor, sample_tensor_x2):
        calculator = TotalVariationL1Loss(is_mean_reduction=True)
        tv_loss = calculator(sample_tensor, sample_tensor_x2)
        assert tv_loss.item() == 0.10999999940395355
