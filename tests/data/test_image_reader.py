"""Tests for image reader


<explanation>
"""

import os

import numpy as np
import pytest
from scripts.data.image_reader import (ImageReader, ImageReaderAsBool,
                                       ImageReaderAsGray, ImageReaderAsY)


class TestImageReader():

    @pytest.fixture()
    def image_reader(self, img_dir_path):
        return ImageReader(img_dir_path)

    def test_load_files(self, image_reader):
        for image in image_reader.load_files():
            assert isinstance(image, np.ndarray)


class TestImageReaderAsGray():

    @pytest.fixture()
    def image_reader(self, img_dir_path):
        return ImageReaderAsGray(img_dir_path)

    def test_load_files(self, image_reader):
        for image in image_reader.load_files():
            assert isinstance(image, np.ndarray)
            assert image.ndim == 2


class TestImageReaderAsY():

    @pytest.fixture()
    def image_reader(self, img_dir_path):
        return ImageReaderAsY(img_dir_path)

    def test_load_files(self, image_reader):
        for image in image_reader.load_files():
            assert isinstance(image, np.ndarray)
            assert image.ndim == 2


class TestImageReaderAsBool():

    @pytest.fixture()
    def image_reader(self, img_dir_path):
        return ImageReaderAsBool(img_dir_path)

    def test_load_files(self, image_reader):
        for image in image_reader.load_files():
            assert isinstance(image, np.ndarray)
            assert image.ndim == 2
            assert image.dtype == 'bool'


@pytest.fixture()
def img_dir_path():
    return os.path.join(os.path.dirname(__file__), '..', 'assets', 'lenna')
