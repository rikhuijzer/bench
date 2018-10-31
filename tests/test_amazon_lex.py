from systems.amazon_lex import *


def test_get_lex_template():
    assert json.loads(get_lex_template())  # json validity test


def test_fill_lex_json():
    assert json.dumps(fill_lex_json(intent='test', utterances=('foo', 'bar')))  # json validity test


def test_get_intents_lex_json():
    corpus = Corpus.WebApplications
    intents = set(map(lambda js: js['resource']['name'], get_intents_lex_json(corpus).json))
    assert get_intents(get_messages(corpus)) == intents
