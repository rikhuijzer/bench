from core.evaluate import *


def test_classify_intents():
    df = classify_intents(System('mock', Corpus.Mock, (14, )), Corpus.Mock).df
    assert 20 == df.shape[0]
    assert 'A' == df.classification[1]
    assert 'C' == df.classification[14]


def test_get_f1_score():
    assert 3 == get_f1_score(System('mock', Corpus.Empty, ()), Corpus.Empty)


def test_get_f1_score_runs():
    result = get_f1_score_runs('mock', Corpus.Empty, n_runs=2)
    assert result[0] != result[1]
