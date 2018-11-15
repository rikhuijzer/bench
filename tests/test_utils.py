import pytest

from src.systems.mock import get_timestamp
from src.utils import get_timestamp, convert_str_timestamp, get_substring_match, add_nested_value, iterate


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
    text = get_timestamp()
    timestamp = convert_str_timestamp(text)
    result_text = str(timestamp)
    assert text == result_text


def test_add_nested_value():
    assert {'a': {'b': {'c': 3}}} == add_nested_value({}, 3, 'a', 'b', 'c')


def test_iterate():
    """Only testing whether iteration is implemented correctly, not whether side-effect has occurred."""
    numbers = range(3)
    iterable = map(lambda number: number + 1, numbers)
    assert iterate(iterable)
