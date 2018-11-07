import src.results
import src.typ
import shutil
import functools

system = src.typ.System('mock', src.typ.Corpus.MOCK, ())
corpus = src.typ.Corpus.MOCK
sc = src.typ.SystemCorpus(system, corpus)
result = src.typ.CSVs.GENERAL


def test_get_filename():
    fn = src.results.get_filename(sc, result)
    assert 'results' == fn.parents[1].name
    assert 'mock-MOCK' == fn.parents[0].name
    assert 'general.yml' == fn.name


def test_write_tuple():
    def clear_cache():
        src.results.create_folder.cache_clear()
        src.results.create_file.cache_clear()

    def create_csv_intent(x: int) -> src.typ.CSVIntent:
        return src.typ.CSVIntent(x, -1, 'sentence', 'intent', 'classification', -1.0, -1)

    csv = src.typ.CSVs.INTENTS

    # write three lines to csv
    for i in range(0, 3):
        if i < 2:
            clear_cache()  # to avoid complexity the cache should not change behaviour, so we test without the cache
        src.results.write_tuple(sc, create_csv_intent(i))

    # check whether the three lines are added
    with open(str(src.results.get_filename(sc, csv)), 'r') as f:
        assert src.results.create_header(create_csv_intent(0)) == f.readline().strip()
        for i in range(0, 3):
            assert src.results.convert_tuple_str(create_csv_intent(i)) == f.readline().strip()

    # since we now have a file with some data we can test functions on this
    assert create_csv_intent(2) == src.results.get_newest_tuple(sc, csv)

    # remove mock-MOCK folder
    shutil.rmtree(str(src.results.get_folder(sc)))


def test_get_tuple_types():
    assert [int, int, str, str, str, float, int] == list(src.results.get_tuple_types(src.typ.CSVIntent))


def test_convert_str_tuple():
    expected = src.typ.CSVIntent(-1, -1, 'sentence', 'intent', 'classification', -1.0, -1)
    t = src.results.convert_str_tuple('-1,-1,sentence,intent,classification,-1.0,-1', src.typ.CSVs.INTENTS)
    assert expected == t
