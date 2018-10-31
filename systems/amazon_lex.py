import json
from typing import Iterable
from typing import NamedTuple, Tuple

from systems.systems import System, Corpus, IntentClassification
from core.training_data import get_train_test, TrainTest, get_intents, get_messages


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


def fill_lex_json(intent: str, utterances: Tuple[str, ...]) -> dict:
    js = json.loads(get_lex_template())
    js['resource']['name'] = intent
    js['resource']['sampleUtterances'] = list(utterances)
    return js


IntentJSON = NamedTuple('IntentJSON', [('intent', str), ('json', dict)])


def get_intents_lex_json(corpus: Corpus) -> Iterable[IntentJSON]:
    """ Generator for Amazon Lex templates each filled for some intent with its utterances. """
    messages = get_train_test(get_messages(corpus), TrainTest.train)
    intents = get_intents(messages)

    for intent in intents:
        utterances = tuple([message.text for message in messages if message.data['intent'] == intent])
        yield IntentJSON(intent, fill_lex_json(intent, utterances))


def train(system: System, corpus: Corpus) -> System:
    return System()


def store_lex_training_data(corpus: Corpus):
    intents = get_intents(get_train_test(get_messages(corpus), TrainTest.train))


def get_intent_lex() -> IntentClassification:
    return IntentClassification()
