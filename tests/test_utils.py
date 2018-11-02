import core.utils
import pytest


def test_get_substring_match():
    pairs = {
        'one': 'foo',
        'two': 'bar'
    }
    assert 'foo' == core.utils.get_substring_match(pairs, 'one-lorem')
    assert 'foo' == core.utils.get_substring_match(pairs, 'one-two')
    with pytest.raises(ValueError):
        core.utils.get_substring_match(pairs, 'three')
