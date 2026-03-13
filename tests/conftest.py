import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (runs bash scripts, not included in default test runs)"
    )
