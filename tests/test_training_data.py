from core.training_data import *

# TODO: Switch to Pytest


def test_tokenizer():
    tokenized = 'How to partially upgrade Ubuntu 11 . 10 from Ubuntu 11 . 04 ? '
    detokenized = 'How to partially upgrade Ubuntu 11.10 from Ubuntu 11.04?'
    assert tokenized == tokenize(detokenized)
    assert detokenized == tokenize(tokenized, detokenize=True)
