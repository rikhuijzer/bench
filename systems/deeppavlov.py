import core.typ
import requests
import systems.systems
import json


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    raise AssertionError('Trying to train {} which should be trained via Docker build'.format(system))


def get_response(query: core.typ.Query) -> core.typ.Response:
    data = {'context': [query.text]}
    r = requests.post('http://localhost:{}/intents'.format(systems.systems.get_port(query.system.name)),
                      data=json.dumps(data), headers=core.typ.Header.json.value)
    return core.typ.Response(r.json()[0][0][0], -1.0, [])
