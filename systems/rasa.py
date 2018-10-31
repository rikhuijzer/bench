from typing import List

import requests
from rasa_nlu.training_data import Message, TrainingData

from core.training_data import get_train_test, get_messages, TrainTest
from systems.systems import System, Corpus, Header, get_port


def train(system: System, corpus: Corpus) -> System:
    training_examples: List[Message] = list(get_train_test(get_messages(corpus), TrainTest.train))
    training_data = TrainingData(training_examples=training_examples).as_json()
    url = 'http://localhost:{}/train?project=my_project'
    r = requests.post(url.format(get_port(system.name)), data=training_data, headers=Header.json.value).json()
    if 'error' in r:
        # print(str(training_data)[344:380].encode("utf-8"))
        raise RuntimeError('Training {} failed on {}, Response: \n {}.'.format(system.name, corpus, r))
    return System(system.name, corpus, ())
