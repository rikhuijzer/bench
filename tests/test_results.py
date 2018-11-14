import src.dataset
import src.results
import src.typ
from src.systems.mock import get_timestamp
from tests.utils import clear_cache, cleanup
from src.results import get_csv_entity
system = src.typ.System('mock', src.typ.Corpus.MOCK, '', ())
corpus = src.typ.Corpus.MOCK
system_corpus = src.typ.SystemCorpus(system, corpus)
result = src.typ.CSVs.GENERAL


def test_get_filename():
    fn = src.results.get_filename(system_corpus, result)
    assert 'results' == fn.parents[1].name
    assert 'mock-MOCK' == fn.parents[0].name
    assert 'general.yml' == fn.name


def test_write_tuple():
    def create_csv_intent(x: int) -> src.typ.CSVIntent:
        return src.typ.CSVIntent(x, get_timestamp(), 'sentence', 'intent', 'classification', -1.0, -1)

    csv = src.typ.CSVs.INTENTS

    def write():
        for i in range(0, 3):
            if i < 2:
                clear_cache()  # to avoid complexity the cache should not change behaviour, so we test without the cache
            src.results.write_tuple(system_corpus, create_csv_intent(i))

    def validate_write():
        with open(str(src.results.get_filename(system_corpus, csv)), 'r') as f:
            assert src.results.create_header(create_csv_intent(0)) == f.readline().strip()
            for i in range(0, 3):
                assert src.results.convert_tuple_str(create_csv_intent(i)) == f.readline().strip()

    def test_get_newest_tuple():
        assert create_csv_intent(2) == src.results.get_newest_tuple(system_corpus, csv)

    def test_get_csv_intent():
        message = src.dataset.create_message('foo', 'bar', [], False, corpus)
        response = src.typ.Response('bar', -1.0, [])
        classification = src.typ.Classification(system_corpus.system, message, response)
        csv_intent = src.results.get_csv_intent(classification)
        expected = src.typ.CSVIntent(id=3, timestamp='', sentence='foo', gold_standard='bar',
                                     classification='bar', confidence=-1.0, time=0)
        assert expected == csv_intent

    write()
    validate_write()
    test_get_newest_tuple()
    test_get_csv_intent()
    cleanup()


def test_get_tuple_types():
    assert [int, str, str, str, str, float, int] == list(src.results.get_tuple_types(src.typ.CSVIntent))


def test_convert_str_tuple():
    expected = src.typ.CSVIntent(-1, 'run', 'sentence', 'intent', 'classification', -1.0, -1)
    t = src.results.convert_str_tuple('-1,run,sentence,intent,classification,-1.0,-1', src.typ.CSVs.INTENTS)
    assert expected == t
