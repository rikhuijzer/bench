import core.typ
import requests
import systems.systems
import json


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    raise AssertionError('Trying to train {} which should be trained via Docker build'.format(system))


def get_intent(system: core.typ.System, test_sentence: core.typ.TestSentence) -> core.typ.IntentClassification:
    data = {'context': [test_sentence.text]}
    r = requests.post('http://localhost:{}/intents'.format(systems.systems.get_port(system.name)),
                      data=json.dumps(data), headers=core.typ.Header.json.value)
    return core.typ.IntentClassification(system, r.json()[0][0][0])
