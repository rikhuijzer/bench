import rasa_nlu.training_data
import requests

import core.training_data
import systems.systems
import core.typ
import json


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    training_examples = list(core.training_data.get_filtered_messages(corpus, core.typ.TrainTest.train))
    training_data = rasa_nlu.training_data.TrainingData(training_examples=training_examples).as_json()
    url = 'http://localhost:{}/train?project=my_project'
    r = requests.post(url.format(systems.systems.get_port(system.name)),
                      data=training_data, headers=core.typ.Header.json.value).json()
    if 'error' in r:
        # print(str(training_data)[344:380].encode("utf-8"))
        raise RuntimeError('Training {} failed on {}, Response: \n {}.'.format(system.name, corpus, r))
    return core.typ.System(system.name, corpus, ())


def get_classification(test_sentence: str) -> core.typ.Classification:
    data = {'q': test_sentence.text, 'project': 'my_project'}
    url = 'http://localhost:{}/parse'
    r = requests.post(url.format(systems.systems.get_port(system.name)),
                      data=json.dumps(data), headers=core.typ.Header.json.value)
    if r.status_code != 200:
        raise RuntimeError('Could not get intent for text: {}'.format(test_sentence))
    return core.typ.IntentClassification(system, r.json()['intent']['name'])
