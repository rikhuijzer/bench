import pytest

import src.utils
from src.systems.mock import timestamp_text


def test_get_substring_match():
    pairs = {
        'one': 'foo',
        'two': 'bar'
    }
    assert 'foo' == src.utils.get_substring_match(pairs, 'one-lorem')
    assert 'foo' == src.utils.get_substring_match(pairs, 'one-two')
    with pytest.raises(ValueError):
        src.utils.get_substring_match(pairs, 'three')


def test_timestamp():
    text = timestamp_text
    timestamp = src.utils.convert_str_timestamp(text)
    result_text = str(timestamp)
    assert text == result_text
