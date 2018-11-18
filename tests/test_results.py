import src.dataset
import src.results
import src.typ as tp
from tests.utils import clear_caches, run_with_file_operations, get_timestamp, get_system_corpus
from typing import Iterable


def create_csv_intent(x: int) -> src.typ.CSVIntent:
    return src.typ.CSVIntent(x, get_timestamp(), 'sentence', 'intent', 'classification', -1.0, -1)


def write_three_distinct_intents(system_corpus: tp.SystemCorpus):
    for i in range(3):
        if i == 2:
            clear_caches()  # to avoid complexity the cache should not change behaviour, so we also clear the cache
        src.results.write_tuple(system_corpus, create_csv_intent(i))


def test_get_filename():
    fn = src.results.get_filename(get_system_corpus('mock'), tp.CSVs.STATS)
    assert 'results' == fn.parents[1].name
    assert 'mock-MOCK' == fn.parents[0].name
    assert 'statistics.yml' == fn.name


def test_write_tuple():
    def helper(sc: tp.SystemCorpus) -> Iterable[str]:
        """Creates file, adds three tuples and returns lines of file as string."""
        write_three_distinct_intents(sc)

        with open(str(src.results.get_filename(sc, tp.CSVs.INTENTS)), 'r') as f:
            yield f.readline().strip()

    result = run_with_file_operations(test_write_tuple.__name__, helper)
    header, *lines = result
    assert src.results.create_header(create_csv_intent(0)) == header
    for i, line in enumerate(lines):
        assert src.results.convert_tuple_str(create_csv_intent(i)) == line


def test_get_newest_tuple():
    def helper(sc: tp.SystemCorpus) -> str:
        """Returns newest tuple from file after adding some data to file."""
        write_three_distinct_intents(sc)
        return src.results.get_newest_tuple(sc, tp.CSVs.INTENTS)

    assert create_csv_intent(2) == run_with_file_operations(test_get_newest_tuple.__name__, helper)


def test_get_csv_intent():
    def helper(sc: tp.SystemCorpus) -> tp.CSVIntent:
        """In get_csv_intent() the newest tuple is obtained from file."""
        write_three_distinct_intents(sc)
        message = src.dataset.create_message('foo', 'bar', [], False, sc.corpus)
        response = tp.Response('bar', -1.0, [])
        classification = tp.Classification(sc.system, message, response)
        return src.results.get_csv_intent(classification)

    expected = tp.CSVIntent(id=3, timestamp=get_timestamp(), sentence='foo', gold_standard='bar',
                            classification='bar', confidence=-1.0, time=0)
    result = run_with_file_operations(test_get_csv_intent.__name__, helper)
    assert expected == result


def test_get_tuple_types():
    assert [int, str, str, str, str, float, int] == list(src.results.get_tuple_types(tp.CSVIntent))


def test_convert_str_tuple():
    expected = tp.CSVIntent(-1, 'run', 'sentence', 'intent', 'classification', -1.0, -1)
    t = src.results.convert_str_tuple('-1,run,sentence,intent,classification,-1.0,-1', tp.CSVs.INTENTS)
    assert expected == t
