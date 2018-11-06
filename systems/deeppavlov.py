import core.typ
import requests
import systems.systems
import json


def train(sc: core.typ.SystemCorpus) -> core.typ.System:
    raise AssertionError('Trying to train {} which should be trained via Docker build'.format(sc.system))


def get_response(query: core.typ.Query) -> core.typ.Response:
    data = {'context': [query.text]}
    r = requests.post('http://localhost:{}/intents'.format(systems.systems.get_port(query.system.name)),
                      data=json.dumps(data), headers=systems.systems.get_header(core.typ.Header.JSON))
    return core.typ.Response(r.json()[0][0][0], -1.0, [])
