import src.typ as tp
from src.evaluate import get_classifications, get_f1_intent, get_statistics
from tests.utils import get_corpus, get_system, run_with_file_operations


def test_classify_intents():
    classifications = tuple(get_classifications(tp.SystemCorpus(get_system(), get_corpus()), retrain=False))
    # classifications are for messages 15, 16, 17, 18, 19
    assert 5 == len(classifications)
    assert 'B' == classifications[0].response.intent
    assert 'C' == classifications[1].response.intent


def test_get_f1_intent():
    result = run_with_file_operations(test_get_f1_intent.__name__, get_f1_intent)
    assert 0.8 == round(result, 1)


def test_get_statistics():
    result = run_with_file_operations(test_get_statistics.__name__, get_statistics)
    assert 'f1 scores' in result


def test_write_statistics():
    assert run_with_file_operations(test_write_statistics.__name__, get_statistics)
