import systems.amazon_lex
import json
import core.typ
import core.training_data


def test_get_lex_template():
    assert json.loads(systems.amazon_lex.get_lex_template())  # json validity test


def test_fill_lex_json():
    assert json.dumps(systems.amazon_lex.fill_lex_json(intent='test', utterances=('foo', 'bar')))  # json validity test


def test_get_intents_lex_json():
    corpus = core.typ.Corpus.WEBAPPLICATIONS
    intents = set(map(lambda int_js: int_js.json['resource']['name'], systems.amazon_lex.get_intents_lex_json(corpus)))
    assert len(intents) > 0
    assert set(core.training_data.get_intents(corpus)) == intents
