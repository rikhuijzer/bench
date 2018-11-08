import src.evaluate
import src.typ
from tests.test_results import cleanup
from tests.utils import system, corpus, system_corpus


def test_classify_intents():
    classifications = tuple(src.evaluate.get_classifications(src.typ.SystemCorpus(system, corpus)))
    # classifications are for messages 15, 16, 17, 18, 19
    assert 5 == len(classifications)
    assert 'B' == classifications[0].response.intent
    assert 'C' == classifications[1].response.intent


def test_run_bench():
    """The run bench calls quite a lot of functions. It will crash if there is a bug."""
    src.evaluate.run_bench(system_corpus)
    cleanup()
