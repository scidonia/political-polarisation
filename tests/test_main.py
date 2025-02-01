import pytest
from political_polarisation.main import *


def test_chunk_tokens():
    chunks = chunk_tokens([1, 2, 3], 2, 1)
    assert chunks == [[1, 2], [2, 3]]
