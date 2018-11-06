import core.results
import core.typ
import shutil
import functools

system = core.typ.System('mock', core.typ.Corpus.MOCK, ())
corpus = core.typ.Corpus.MOCK
sc = core.typ.SystemCorpus(system, corpus)
result = core.typ.CSVs.GENERAL


def test_get_filename():
    fn = core.results.get_filename(sc, result)
    assert 'results' == fn.parents[1].name
    assert 'mock-MOCK' == fn.parents[0].name
    assert 'general.yml' == fn.name


def test_write_tuple():
    def clear_cache():
        core.results.create_folder.cache_clear()
        core.results.create_file.cache_clear()

    def create_csv_intent(x: int) -> core.typ.CSVIntent:
        return core.typ.CSVIntent(x, -1, 'sentence', 'intent', 'classification', -1.0, -1)

    csv = core.typ.CSVs.INTENTS

    # write three lines to csv
    for i in range(0, 3):
        if i < 2:
            clear_cache()  # to avoid complexity the cache should not change behaviour, so we test without the cache
        core.results.write_tuple(sc, create_csv_intent(i))

    # check whether the three lines are added
    with open(str(core.results.get_filename(sc, csv)), 'r') as f:
        assert core.results.create_header(create_csv_intent(0)) == f.readline().strip()
        for i in range(0, 3):
            assert core.results.convert_tuple_str(create_csv_intent(i)) == f.readline().strip()

    # since we now have a file with some data we can test functions on this
    assert create_csv_intent(2) == core.results.get_newest_tuple(sc, csv)

    # remove mock-MOCK folder
    shutil.rmtree(str(core.results.get_folder(sc)))


def test_get_tuple_types():
    assert [int, int, str, str, str, float, int] == list(core.results.get_tuple_types(core.typ.CSVIntent))


def test_convert_str_tuple():
    expected = core.typ.CSVIntent(-1, -1, 'sentence', 'intent', 'classification', -1.0, -1)
    t = core.results.convert_str_tuple('-1,-1,sentence,intent,classification,-1.0,-1', core.typ.CSVs.INTENTS)
    assert expected == t
