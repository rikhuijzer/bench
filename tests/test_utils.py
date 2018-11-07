import src.utils
import pytest


def test_get_substring_match():
    pairs = {
        'one': 'foo',
        'two': 'bar'
    }
    assert 'foo' == src.utils.get_substring_match(pairs, 'one-lorem')
    assert 'foo' == src.utils.get_substring_match(pairs, 'one-two')
    with pytest.raises(ValueError):
        src.utils.get_substring_match(pairs, 'three')
