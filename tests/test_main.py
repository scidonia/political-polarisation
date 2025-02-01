import pytest
import political_polarisation.main as main
import tiktoken
import pyarrow as pa
import numpy


def test_chunk_tokens():
    chunks = main.chunk_tokens([1, 2, 3], 2, 1)
    assert chunks == [[1, 2], [2, 3]]


def test_short_chunk_tokens():
    chunks = main.chunk_tokens([1, 2, 3], 500, 10)
    assert chunks == [[1, 2, 3]]


def test_mechanical_clean_chunk():
    enc: tiktoken.Encoding = tiktoken.get_encoding("o200k_base")
    sentence = """ name was Bill.
    This is an example with garbage at the beginning and at the end.
    And there was no
    """
    tokens = enc.encode(sentence)
    cleaned = main.mechanical_clean_chunk(enc, tokens)
    result = enc.decode(cleaned)
    assert result == "This is an example with garbage at the beginning and at the end."


def test_oracular_text_chunking(monkeypatch):
    def ask_oracle_for_chunk_mock(enc, tokens):
        return [1, 2, 3, 4]

    monkeypatch.setattr(main, "ask_oracle_for_chunk", ask_oracle_for_chunk_mock)
    records = pa.array(["This is a text chunker.", "And it is chunking well."])
    result = main.oracular_text_chunking(records).to_pylist()
    expected = [['"#$%'], ['"#$%']]
    assert result == expected
