"""Tests of mch


<explanation>
"""

import pytest
import torch
from scripts.model.mch import MCH


@pytest.mark.parametrize(('scale', 'input_size', 'out_size'),
                         [(2, [32, 1, 30, 30], [32, 1, 40, 40]),
                          (4, [32, 1, 30, 30], [32, 1, 96, 96])])
def test_mch_initialization(gpu_device, scale, input_size, out_size):
    net = MCH(scale, 1)
    gpu_net = net.to(gpu_device)

    tensor = torch.ones(*input_size)
    tensor_gpu = tensor.to(gpu_device)

    cnn_output = gpu_net(tensor_gpu)

    assert cnn_output.shape == torch.Size(out_size)
    torch.cuda.empty_cache()
