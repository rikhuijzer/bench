import rasa_nlu.training_data
import requests

import core.training_data
import systems.systems
import core.typ
import json


def train(sc: core.typ.SystemCorpus) -> core.typ.System:
    training_examples = list(core.training_data.get_filtered_messages(sc.corpus, train=True))
    training_data = rasa_nlu.training_data.TrainingData(training_examples=training_examples).as_json()
    url = 'http://localhost:{}/train?project=my_project'
    r = requests.post(url.format(systems.systems.get_port(sc.system.name)),
                      data=training_data, headers=systems.systems.get_header(core.typ.Header.JSON)).json()
    if 'error' in r:
        # print(str(training_data)[344:380].encode("utf-8"))
        raise RuntimeError('Training {} failed on {}, Response: \n {}.'.format(sc.system.name, sc.corpus, r))
    return core.typ.System(sc.system.name, sc.corpus, ())


def get_response(query: core.typ.Query) -> core.typ.Response:
    data = {'q': query.text, 'project': 'my_project'}
    url = 'http://localhost:{}/parse'
    r = requests.post(url.format(systems.systems.get_port(query.system.name)),
                      data=json.dumps(data), headers=systems.systems.get_header(core.typ.Header.JSON))
    if r.status_code != 200:
        raise RuntimeError('Could not get intent for text: {}'.format(query.text))
    return core.typ.Response(r.json()['intent']['name'], '-1.0', [])
