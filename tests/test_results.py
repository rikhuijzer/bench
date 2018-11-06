import core.results
import core.typ
import shutil

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

    csv_intent = core.typ.CSVIntent(-1, -1, 'sentence', 'intent', 'classification', -1.0, -1)

    for _ in range(0, 3):
        clear_cache()  # to avoid complexity the cache should not change behaviour, so we test without the cache
        core.results.write_tuple(sc, csv_intent)

    with open(str(core.results.get_filename(sc, core.typ.CSVs.INTENTS)), 'r') as f:
        assert core.results.create_header(csv_intent) == f.readline().strip()
        for _ in range(0, 3):
            assert core.results.convert_tuple_str(csv_intent) == f.readline().strip()

    shutil.rmtree(str(core.results.get_folder(sc)))


def test_get_tuple_types():
    assert [int, int, str, str, str, float, int] == list(core.results.get_tuple_types(core.typ.CSVIntent))


def test_convert_str_tuple():
    expected = core.typ.CSVIntent(-1, -1, 'sentence', 'intent', 'classification', -1.0, -1)
    t = core.results.convert_str_tuple('-1,-1,sentence,intent,classification,-1.0,-1', core.typ.CSVs.INTENTS)
    assert expected == t
