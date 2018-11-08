import json
import typing
import src.typ
import src.training_data


def get_lex_template() -> str:
    """ Default template for Amazon Lex Intents. Priority set to 5, should mean low according to documentation. """
    return '''
    {
        "metadata": {
            "schemaVersion": "1.0",
            "importType": "LEX",
            "importFormat": "JSON"
        },
        "resource": {
            "description": "intent description",
            "name": "intent name",
            "version": 1,
            "fulfillmentActivity": {
                "type": "ReturnIntent"
            },
            "sampleUtterances": [
                "string",
                "string"
            ],
            "slots": [],
            "slotTypes": []
        }
    }
    '''


def fill_lex_json(intent: str, utterances: typing.Tuple[str, ...]) -> dict:
    js = json.loads(get_lex_template())
    js['resource']['name'] = intent
    js['resource']['sampleUtterances'] = list(utterances)
    return js


IntentJSON = typing.NamedTuple('IntentJSON', [('intent', str), ('json', dict)])


def get_intents_lex_json(corpus: src.typ.Corpus) -> typing.Iterable[IntentJSON]:
    """ Generator for Amazon Lex templates each filled for some intent with its utterances. """
    messages = src.training_data.get_filtered_messages(corpus, train=True)
    intents = src.training_data.get_intents(corpus)

    for intent in intents:
        utterances = tuple([message.text for message in messages if message.data['intent'] == intent])
        yield IntentJSON(intent, fill_lex_json(intent, utterances))


def train(system_corpus: src.typ.SystemCorpus) -> src.typ.System:
    return src.typ.System()


def store_lex_training_data(corpus: src.typ.Corpus):
    intents = set(src.training_data.get_intents(
        src.training_data.get_filtered_messages(src.training_data.get_messages(corpus), train=True)))


def get_response(query: src.typ.Query) -> src.typ.Response:
    return src.typ.Response()
