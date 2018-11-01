import core.evaluate
import core.typ


def test_classify_intents():
    df = core.evaluate.classify_intents(core.typ.System('mock', core.typ.Corpus.Mock, (14,)), core.typ.Corpus.Mock).df
    assert 20 == df.shape[0]
    assert 'A' == df.classification[1]
    assert 'C' == df.classification[14]


def test_get_f1_score():
    mock = core.typ.Corpus.Mock
    assert 0.5 == core.evaluate.get_f1_score(core.typ.System('mock', core.typ.Corpus.Empty, (2,)), mock).scores[0]
    assert 0.6 == core.evaluate.get_f1_score(core.typ.System('mock', mock, (3,)), mock).scores[0]


def test_get_f1_score_runs():
    mock = core.typ.Corpus.Mock
    result = core.evaluate.get_f1_score_runs(core.typ.System('mock', mock, (3,)), mock, n_runs=2)
    assert result[0] != result[1]
