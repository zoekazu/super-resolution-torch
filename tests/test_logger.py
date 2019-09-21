"""Tests for logger


<explanation>
"""

from scripts.logger import get_logger


def test_get_logger():
    logger = get_logger(__file__)
    logger.info('example')
