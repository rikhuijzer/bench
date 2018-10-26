import requests
import yaml

from core.training_data import *
from core.utils import *


class Header(Enum):
    json = {'content-type': 'application/json'}
    yml = {'content-type': 'application/x-yml'}


@lru_cache(maxsize=1)
def get_docker_compose_configuration() -> dict:
    with open(Path(__file__).parent.parent / 'docker-compose.yml', 'rb') as f:
        return yaml.load(f)


def get_n_systems() -> int:
    return len(get_docker_compose_configuration()['services'])


def get_port(system: str) -> int:
    return int(get_docker_compose_configuration()['services'][system]['ports'][0][0:4])


@lru_cache(maxsize=square_ceil(get_n_systems()))
def train(system: str, corpus: Corpus) -> bool:
    if 'rasa' in system:
        training_examples: List[Message] = list(get_train_test(get_messages(corpus), TrainTest.train))
        training_data = TrainingData(training_examples=training_examples).as_json()
        url = 'http://localhost:{}/train?project=my_project'
        r = requests.post(url.format(get_port(system)), data=training_data, headers=Header.json.value).json()
        if 'error' in r:
            raise RuntimeError('Training of system: {} failed. Corpus: {}, Response: \n {}.'.format(system, corpus, r))
        return True

    elif 'deeppavlov' in system:
        return True

    raise ValueError('Unknown system: {}.'.format(system))


def get_intent(system: str, text: str, corpus: Corpus = None) -> str:
    train(system, corpus)

    if 'rasa' in system:
        data = {'q': text, 'project': 'my_project'}
        url = 'http://localhost:{}/parse'
        r = requests.post(url.format(get_port(system)), data=json.dumps(data), headers=Header.json.value)

        if r.status_code != 200:
            raise RuntimeError('Could not get intent for text: {}'.format(text))
        return r.json()['intent']['name']

    elif 'deeppavlov' in system:
        data = {'context': [text]}
        r = requests.post('http://localhost:{}/intents'.format(get_port(system)), data=json.dumps(data),
                          headers=Header.json.value)
        return r.json()[0][0][0]

    elif 'watson' in system:
        raise AssertionError('Not implemented. Possibly interesting: https://github.com/joe4k/wdcutils/')
