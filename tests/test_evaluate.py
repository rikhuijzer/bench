import src.evaluate
import src.typ


def test_classify_intents():
    system = src.typ.System('mock', src.typ.Corpus.MOCK, '', (16,))
    corpus = src.typ.Corpus.MOCK
    classifications = tuple(src.evaluate.get_classifications(src.typ.SystemCorpus(system, corpus)))
    # classifications are for messages 15, 16, 17, 18, 19
    assert 5 == len(classifications)
    assert 'B' == classifications[0].response.intent
    assert 'C' == classifications[1].response.intent
