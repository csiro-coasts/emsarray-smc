import pathlib

import pytest


@pytest.fixture
def datasets() -> pathlib.Path:
    here = pathlib.Path(__file__).parent
    return here / 'datasets'
