import pytest
import matplotlib


def pytest_configure(config):
    matplotlib.use("Agg")


def pytest_ignore_collect(collection_path, config):
    if collection_path.name == "test_greek_mixin_lru_cache.py":
        return True
