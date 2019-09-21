"""Tests for dataset loader


<explanation>
"""

import os

import pytest
import torch
from scripts.data.dataset_loader import (ValidateDataset, TrainDataset,
                                         TrainDatasetParam, recursive_len_list)
from scripts.data.image_reader import ImageReader
from torchvision.transforms import transforms


class TestTestDataset():

    @pytest.mark.parametrize(('scale', 'shave_size'),
                             [(2, 5)])
    def test_len(self, img_dir_path, scale, shave_size):
        reader = ImageReader(img_dir_path)
        loader = ValidateDataset(reader, scale, shave_size)
        assert len(loader) == 3


class TestTrainDatasetParam():

    @pytest.mark.parametrize(('stride_size', 'shave_size', 'scale', 'label_size', 'input_size',
                              'ideal_label_size', 'ideal_input_size'),
                             [(15, 0, 2, None, 30, 60, 30),
                              (15, 2, 2, None, 30, 52, 30),
                              (15, 0, 2, 60, None, 60, 30),
                              (15, 2, 2, 52, None, 52, 30),
                              (15, 0, 4, None, 15, 60, 15),
                              (15, 2, 4, None, 15, 44, 15),
                              (15, 0, 4, 60, None, 60, 15),
                              (15, 2, 4, 44, None, 44, 15)])
    def test_correct_args(self, stride_size, shave_size, scale, label_size, input_size,
                          ideal_label_size, ideal_input_size):
        param = TrainDatasetParam(stride_size=stride_size,
                                  shave_size=shave_size,
                                  scale=scale,
                                  label_size=label_size,
                                  input_size=input_size)
        assert param.input_size == ideal_input_size
        assert param.label_size == ideal_label_size

    @pytest.mark.xfail
    @pytest.mark.parametrize(('stride_size', 'shave_size', 'scale', 'label_size', 'input_size'),
                             [(15, 10, 2, 30, 30)])
    def test_fault_args(self, stride_size, shave_size, scale, label_size, input_size):
        TrainDatasetParam(stride_size=stride_size,
                          shave_size=shave_size,
                          scale=scale,
                          label_size=label_size,
                          input_size=input_size)


class TestTrainDataset():

    @pytest.fixture()
    def image_reader(self, img_dir_path):
        return ImageReader(img_dir_path)

    @pytest.fixture(params=[True, False])
    def transformer(self, request):
        if request.param:
            return transforms.Compose([transforms.ToTensor()])
        return None

    @pytest.fixture(params=[({'stride_size': 15, 'shave_size': 2, 'scale': 2, 'label_size': 52},
                             {'batch_size': 1, 'len': 2118}),
                            ({'stride_size': 15, 'shave_size': 2, 'scale': 2, 'input_size': 30},
                             {'batch_size': 1, 'len': 2118}),
                            ({'stride_size': 15, 'shave_size': 2, 'scale': 2, 'label_size': 52},
                             {'batch_size': 32, 'len': 67}),
                            ({'stride_size': 15, 'shave_size': 2, 'scale': 2, 'input_size': 30},
                             {'batch_size': 32, 'len': 67})])
    def loader_param(self, request):
        return TrainDatasetParam(**request.param[0]), request.param[1]

    def test_loader_len(self, image_reader, loader_param, transformer):
        dataset = TrainDataset(image_reader, loader_param[0], transformer)
        loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=loader_param[1]['batch_size'], shuffle=True)
        assert len(loader) == loader_param[1]['len']


@pytest.mark.parametrize('input, output',
                         [([[1, 1, 1], [2, 2, 2]], 6),
                          ([[{'key': 'value'}, {'key': 'value'}],
                            [{'key': 'value'}, {'key': 'value'}]], 4)])
def test_recursive_len_list(input, output):
    assert recursive_len_list(input) == output


@pytest.fixture()
def img_dir_path():
    return os.path.join(os.path.dirname(__file__), '..', 'assets', 'lenna')
