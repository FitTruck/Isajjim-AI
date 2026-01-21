"""
pytest configuration for async tests
"""
import pytest


# pytest-asyncio mode 설정
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "asyncio: marks tests as async")
