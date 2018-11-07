import rasa_nlu.training_data
import requests

import src.training_data
import src.typ
import json


# example output: https://rasa.com/docs/nlu/0.13.7/choosing_pipeline/#choosing-pipeline


def train(sc: src.typ.SystemCorpus) -> src.typ.System:
    training_examples = list(src.training_data.get_filtered_messages(sc.corpus, train=True))
    training_data = rasa_nlu.training_data.TrainingData(training_examples=training_examples).as_json()
    url = 'http://localhost:{}/train?project=my_project'
    r = requests.post(url.format(src.system.get_port(sc.system.name)),
                      data=training_data, headers=src.system.get_header(src.typ.Header.JSON)).json()
    if 'error' in r:
        # print(str(training_data)[344:380].encode("utf-8"))
        raise RuntimeError('Training {} failed on {}, Response: \n {}.'.format(sc.system.name, sc.corpus, r))
    return src.typ.System(sc.system.name, sc.corpus, ())


def get_response(query: src.typ.Query) -> src.typ.Response:
    data = {'q': query.text, 'project': 'my_project'}
    url = 'http://localhost:{}/parse'
    r = requests.post(url.format(src.system.get_port(query.system.name)),
                      data=json.dumps(data), headers=src.system.get_header(src.typ.Header.JSON))
    if r.status_code != 200:
        raise RuntimeError('Could not get intent for text: {}'.format(query.text))
    return src.typ.Response(r.json()['intent']['name'], '-1.0', [])
