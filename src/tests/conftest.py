"""
src/tests/conftest.py

Author: Jared Moore
Date: July, 2025

Sets up test markers
"""

import os

import pytest

# Avoid network fetches during litellm import in tests.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def pytest_addoption(parser):
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="run expensive API tests (those marked @pytest.mark.expensive)",
    )


def pytest_configure(config):
    # register the marker so pytest won’t warn about unknown markers
    config.addinivalue_line(
        "markers", "expensive: mark test as expensive (requires real API calls)"
    )


@pytest.fixture(autouse=True)
def _skip_expensive_marker(request):
    """
    Automatically skip any test marked with @pytest.mark.expensive
    unless --run-expensive was given.
    """
    if request.node.get_closest_marker("expensive") and not request.config.getoption(
        "--run-expensive"
    ):
        pytest.skip("Skipping expensive test; use --run-expensive to enable")
