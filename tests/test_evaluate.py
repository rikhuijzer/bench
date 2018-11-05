import core.evaluate
import core.typ


def test_classify_intents():
    system = core.typ.System('mock', core.typ.Corpus.Mock, (16,))
    corpus = core.typ.Corpus.Mock
    classifications = tuple(core.evaluate.get_classifications(core.typ.SystemCorpus(system, corpus)))
    # classifications are for messages 15, 16, 17, 18, 19
    assert 5 == len(classifications)
    assert 'B' == classifications[0].response.intent
    assert 'C' == classifications[1].response.intent
