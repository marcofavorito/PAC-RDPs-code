"""Conftest module."""

import shutil
import sys
import tempfile
from contextlib import contextmanager

import pytest

# this is to import fixtures
from tests.pdfas import pdfa_one_state, pdfa_two_states  # noqa: E402, F401


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


@pytest.fixture(params=["pdfa_one_state"])
def pdfas(request):
    """Get a list of PDFAs."""
    return request.getfuncargvalue(request.param)
