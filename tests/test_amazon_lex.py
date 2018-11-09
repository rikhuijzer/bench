import src.systems.amazon_lex
import json
import src.typ
import src.dataset


def test_get_lex_template():
    assert json.loads(src.systems.amazon_lex.get_lex_template())  # json validity test


def test_fill_lex_json():
    assert json.dumps(src.systems.amazon_lex.fill_lex_json(intent='test', utterances=('foo', 'bar')))  # json validity test


def test_get_intents_lex_json():
    corpus = src.typ.Corpus.WEBAPPLICATIONS
    intents = set(map(lambda int_js: int_js.json['resource']['name'], src.systems.amazon_lex.get_intents_lex_json(corpus)))
    assert len(intents) > 0
    assert set(src.dataset.get_intents(corpus)) == intents
