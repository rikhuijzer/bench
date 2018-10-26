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
    if system == 'rasa-spacy':
        training_examples: List[Message] = list(get_train_test(get_messages(corpus), TrainTest.train))
        training_data = TrainingData(training_examples=training_examples).as_json()
        url = 'http://localhost:{}/train?project=my_project'
        r = requests.post(url.format(get_port(system)), data=training_data, headers=Header.json.value).json()
        if 'error' in r:
            raise RuntimeError('Training of system: {} failed. Corpus: {}, Response: \n {}.'.format(system, corpus, r))
        return True
    raise ValueError('Unknown system: {}.'.format(system))


def get_intent(system: str, corpus: Corpus, text: str) -> str:
    train(system, corpus)

    if system == 'rasa-spacy':
        data = {'q': text, 'project': 'my_project'}
        url = 'http://localhost:{}/parse'
        r = requests.post(url.format(get_port(system)), data=json.dumps(data), headers=Header.json.value)

        if r.status_code != 200:
            raise RuntimeError('Could not get intent for text: {}'.format(text))
        return r.json()['intent']['name']


'''
class Rasa(System):
    trained = False

    def __init__(self, port: int):
        super().__init__(port)

    def train(self):
        print('Training Rasa...')
        # data = {'q': 'b'}
        training_data: TrainingData = TrainingData(training_examples=training_examples)
        url = 'http://localhost:{}/train?project=my_project'
        r = requests.post(url.format(self.port), data=json.dumps(data), headers=Headers.json).json()
        print('Response: ' + str(r))
        if 'error' in r:
            raise RuntimeError('Training failed.')
        self.trained = True

    def get_intent(self, sentence: str):
        if not self.trained:
            self.train()

        data = {'q': sentence}
        url = 'http://localhost:{}/parse'
        r = requests.post(url.format(self.port), data=json.dumps(data), headers=Headers.json)
        return r.json()['intent']['name']


class DeepPavlov(System):
    def __init__(self, port: int):
        super().__init__(port)

    def get_intent(self, sentence: str):
        data = {'context': [sentence]}
        r = requests.post('http://localhost:{}/intent'.format(self.port), data=json.dumps(data),
                          headers=Headers.json)
        return r.json()[0][0][0]


class Watson(System):
    # possibly interesting: https://github.com/joe4k/wdcutils/
    def __init__(self, url):
        super().__init__(url)

    def get_intent(self, sentence: str):
        print('unimplemented')
'''
