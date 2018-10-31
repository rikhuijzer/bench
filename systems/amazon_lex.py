from core.training_data import *
from typing import Iterable


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


def get_intents_lex_json(corpus: Corpus) -> Iterable[dict]:
    """ Generator for Amazon Lex templates each filled for some intent with its utterances. """
    messages = get_train_test(get_messages(corpus), TrainTest.train)
    intents = get_intents(messages)

    for intent in intents:
        utterances = tuple([message.text for message in messages if message.data['intent'] == intent])
        yield fill_lex_json(intent, utterances)
