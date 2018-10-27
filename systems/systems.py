import requests
import yaml

from core.training_data import *
from typing import NamedTuple


class Header(Enum):
    json = {'content-type': 'application/json'}
    yml = {'content-type': 'application/x-yml'}


System = NamedTuple('System', [('name', str), ('knowledge', Corpus)])


@lru_cache(maxsize=1)
def get_docker_compose_configuration() -> dict:
    with open(Path(__file__).parent.parent / 'docker-compose.yml', 'rb') as f:
        return yaml.load(f)


def get_n_systems() -> int:
    return len(get_docker_compose_configuration()['services'])


def get_port(system: str) -> int:
    return int(get_docker_compose_configuration()['services'][system]['ports'][0][0:4])


def train(name: str, corpus: Corpus) -> System:
    if 'rasa' in name:
        training_examples: List[Message] = list(get_train_test(get_messages(corpus), TrainTest.train))
        training_data = TrainingData(training_examples=training_examples).as_json()
        url = 'http://localhost:{}/train?project=my_project'
        r = requests.post(url.format(get_port(name)), data=training_data, headers=Header.json.value).json()
        if 'error' in r:
            raise RuntimeError('Training of system: {} failed. Corpus: {}, Response: \n {}.'.format(name, corpus, r))
        return System(name, corpus)

    raise ValueError('Unknown system for training: {}.'.format(name))


def get_intent(system: System, test_sentence: TestSentence) -> str:
    if test_sentence.corpus != system.knowledge:
        system = train(system.name, test_sentence.corpus)

    if 'rasa' in system.name:
        data = {'q': test_sentence.text, 'project': 'my_project'}
        url = 'http://localhost:{}/parse'
        r = requests.post(url.format(get_port(system.name)), data=json.dumps(data), headers=Header.json.value)

        if r.status_code != 200:
            raise RuntimeError('Could not get intent for text: {}'.format(test_sentence.text))
        return r.json()['intent']['name']

    elif 'deeppavlov' in system.name:
        data = {'context': [test_sentence.text]}
        r = requests.post('http://localhost:{}/intents'.format(get_port(system.name)), data=json.dumps(data),
                          headers=Header.json.value)
        return r.json()[0][0][0]

    elif 'watson' in system:
        raise AssertionError('Not implemented. Possibly interesting: https://github.com/joe4k/wdcutils/')
