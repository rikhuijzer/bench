from src.evaluate import get_classifications, get_f1_intent, run_bench
import src.typ as tp
from tests.test_results import cleanup
from tests.utils import get_corpus, get_system, get_system_corpus


def test_classify_intents():
    classifications = tuple(get_classifications(tp.SystemCorpus(get_system(), get_corpus()), retrain=False))
    # classifications are for messages 15, 16, 17, 18, 19
    assert 5 == len(classifications)
    assert 'B' == classifications[0].response.intent
    assert 'C' == classifications[1].response.intent


def test_run_bench():
    """The run bench calls quite a lot of functions. It will crash if there is a bug."""
    name = 'mock_' + test_run_bench.__name__
    tuple(run_bench(get_system_corpus(name)))
    cleanup(name)


def test_get_f1_intent():
    name = 'mock_' + test_get_f1_intent.__name__
    system_corpus = get_system_corpus(name)
    tuple(run_bench(system_corpus))
    assert 0.8 == get_f1_intent(system_corpus)
    cleanup(name)
