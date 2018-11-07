import src.typ
import requests
import json


def train(sc: src.typ.SystemCorpus) -> src.typ.System:
    raise AssertionError('Trying to train {} which should be trained via Docker build'.format(sc.system))


def get_response(query: src.typ.Query) -> src.typ.Response:
    data = {'context': [query.text]}
    r = requests.post('http://localhost:{}/intents'.format(src.system.get_port(query.system.name)),
                      data=json.dumps(data), headers=src.system.get_header(src.typ.Header.JSON))
    return src.typ.Response(r.json()[0][0][0], -1.0, [])
