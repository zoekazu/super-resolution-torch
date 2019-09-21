"""conftest for pytest


<explanation>
"""

import pytest


@pytest.fixture
def gpu_device():
    return 'cuda'
