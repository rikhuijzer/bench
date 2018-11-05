import core.write
import core.typ
import shutil

system = core.typ.System('mock', core.typ.Corpus.Mock, ())
corpus = core.typ.Corpus.Mock
sc = core.typ.SystemCorpus(system, corpus)
result = core.typ.CSVs.General


def test_get_filename():
    fn = core.write.get_filename(sc, result)
    assert 'results' == fn.parents[1].name
    assert 'mock-Mock' == fn.parents[0].name
    assert 'general.yml' == fn.name


def test_write_tuple():
    csv_intent = core.typ.CSVIntent(-1, -1, 'sentence', 'intent', 'classification', -1.0, -1)

    for _ in range(0, 3):
        core.write.write_tuple(sc, csv_intent)

    with open(str(core.write.get_filename(sc, core.typ.CSVs.Intents)), 'r') as f:
        assert core.write.create_header(csv_intent) == f.readline().strip()
        for _ in range(0, 3):
            assert core.write.convert_tuple(csv_intent) == f.readline().strip()

    # remove 'mock-Mock'  folder recursively
    shutil.rmtree(str(core.write.get_folder(sc)))
