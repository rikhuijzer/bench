import json
import typing
import core.typ
import core.training_data


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


def get_intents_lex_json(corpus: core.typ.Corpus) -> typing.Iterable[IntentJSON]:
    """ Generator for Amazon Lex templates each filled for some intent with its utterances. """
    messages = core.training_data.get_filtered_messages(corpus, core.typ.TrainTest.train)
    intents = core.training_data.get_intents(corpus, core.typ.TrainTest.train)

    for intent in intents:
        utterances = tuple([message.text for message in messages if message.data['intent'] == intent])
        yield IntentJSON(intent, fill_lex_json(intent, utterances))


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    return core.typ.System()


def store_lex_training_data(corpus: core.typ.Corpus):
    intents = core.training_data.get_intents(
        core.training_data.get_train_test(core.training_data.get_messages(corpus), core.typ.TrainTest.train))


def get_intent_lex() -> core.typ.IntentClassification:
    return core.typ.IntentClassification()
