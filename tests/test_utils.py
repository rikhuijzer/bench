import pytest

from src.systems.mock import timestamp_text
from src.utils import get_timestamp, convert_str_timestamp, get_substring_match


def test_get_substring_match():
    pairs = {
        'one': 'foo',
        'two': 'bar'
    }
    assert 'foo' == get_substring_match(pairs, 'one-lorem')
    assert 'foo' == get_substring_match(pairs, 'one-two')
    with pytest.raises(ValueError):
        get_substring_match(pairs, 'three')


def test_get_timestamp():
    convert_str_timestamp(get_timestamp())


def test_timestamp():
    text = timestamp_text
    timestamp = convert_str_timestamp(text)
    result_text = str(timestamp)
    assert text == result_text
