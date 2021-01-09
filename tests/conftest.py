"""Conftest module."""
import logging
import multiprocessing
import shutil
import sys
import tempfile
from contextlib import contextmanager

import pytest

logger = logging.getLogger(__name__)


def get_nb_processes():
    """Get number of processes to use."""
    result = multiprocessing.cpu_count()
    return result


@pytest.fixture(scope="session")
def nb_processes():
    """Get the number of processes available."""
    result = get_nb_processes()
    logger.debug(f"Number of cpus: {result}")
    return result


@contextmanager
def tempdir():
    """Create a temporary directory as a context manager."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except IOError:
            sys.stderr.write("Failed to clean up temp dir {}".format(path))
