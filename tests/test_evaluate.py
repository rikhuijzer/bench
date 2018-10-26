from core.evaluate import *


def test_classify_intents():
    """ TODO: Improve tests by creating a deterministic DeepPavlov bot instance (see Github README for DeepPavlov). """
    corpus = Corpus.WebApplications
    df = classify_intents(corpus, 'rasa-spacy')
    assert df.shape[0] == len(get_train_test(get_messages(corpus), TrainTest.test))


def test_get_f1_score():
    assert 0.0 < get_f1_score(Corpus.WebApplications, 'rasa-spacy') < 1.0


def test_get_f1_score_runs():
    result = get_f1_score_runs(Corpus.WebApplications, 'rasa-spacy', n_runs=2)
    assert result[0] != result[1]
