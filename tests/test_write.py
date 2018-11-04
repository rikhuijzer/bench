import core.write
import core.typ

system = core.typ.System('mock', core.typ.Corpus.Mock, ())
corpus = core.typ.Corpus.Mock
result = core.typ.Result.General


def test_get_filename():
    fn = core.write.get_filename(system, corpus, result)
    assert 'results' == fn.parents[1].name
    assert 'mock-Mock' == fn.parents[0].name
    assert 'general.yml' == fn.name


def test_append_text():
    # create some fictional files and write something to all of them, then delete
    fn = core.write.get_filename(system, corpus, result)
    core.write.append_text('test', fn)
