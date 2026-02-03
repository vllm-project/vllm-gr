import pytest


def pytest_addoption(parser):
    parser.addoption("--loops", action="store", default=1, type=int, help="number to test loops")


@pytest.fixture
def loops(request) -> int:
    return request.config.getoption("--loops")
