import core.typ
import requests
import systems.systems
import json


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    raise AssertionError('Trying to train {} which should be trained via Docker build'.format(system))


def get_classification(test_sentence: str) -> core.typ.Classification:
    data = {'context': [test_sentence.text]}
    r = requests.post('http://localhost:{}/intents'.format(systems.systems.get_port(system.name)),
                      data=json.dumps(data), headers=core.typ.Header.json.value)
    return core.typ.Classification(r.json()[0][0][0], -1.0, [])
