import requests
import yaml

from core.training_data import *
from typing import NamedTuple


class Header(Enum):
    json = {'content-type': 'application/json'}
    yml = {'content-type': 'application/x-yml'}


System = NamedTuple('System', [('name', str), ('knowledge', Corpus), ('data', Tuple)])
IntentClassification = NamedTuple('IntentClassification', [('system', System), ('classification', str)])


@lru_cache(maxsize=1)
def get_docker_compose_configuration() -> dict:
    with open(Path(__file__).parent.parent / 'docker-compose.yml', 'rb') as f:
        return yaml.load(f)


def get_n_systems() -> int:
    return len(get_docker_compose_configuration()['services'])


def get_port(system: str) -> int:
    return int(get_docker_compose_configuration()['services'][system]['ports'][0][0:4])


def train(system: System, corpus: Corpus) -> System:
    """ Train system on corpus. """
    print('Training {} on {}...'.format(system, corpus))

    if 'mock' == system.name:
        data = list(system.data)
        data[0] += 1
        return System(system.name, corpus, tuple(data))

    if 'rasa' in system.name:
        training_examples: List[Message] = list(get_train_test(get_messages(corpus), TrainTest.train))
        training_data = TrainingData(training_examples=training_examples).as_json()
        url = 'http://localhost:{}/train?project=my_project'
        r = requests.post(url.format(get_port(system.name)), data=training_data, headers=Header.json.value).json()
        if 'error' in r:
            raise RuntimeError('Training {} failed on {}, Response: \n {}.'.format(system.name, corpus, r))
        return System(system.name, corpus, ())

    if 'deeppavlov' in system.name:
        raise AssertionError('Training requested for {} which should not be trained.'.format(system.name))

    raise ValueError('Unknown system for training: {}.'.format(system.name))


def get_intent(system: System, test_sentence: TestSentence) -> IntentClassification:
    """ Get intent for some system and some sentence. Function will train system if that is necessary. """
    if test_sentence.corpus != system.knowledge or 'retrain' in system.data:
        system = System(system.name, system.knowledge, tuple(filter(lambda x: x != 'retrain', system.data)))
        system = train(system, test_sentence.corpus)

    if 'mock' == system.name:
        value = int(test_sentence.text)
        classifications = 'A' if (0 < value <= 10) else 'B'
        return IntentClassification(system, 'C' if (value % system.data[0] == 0) else classifications)

    if 'rasa' in system.name:
        data = {'q': test_sentence.text, 'project': 'my_project'}
        url = 'http://localhost:{}/parse'
        r = requests.post(url.format(get_port(system.name)), data=json.dumps(data), headers=Header.json.value)

        if r.status_code != 200:
            raise RuntimeError('Could not get intent for text: {}'.format(test_sentence.text))
        return IntentClassification(system, r.json()['intent']['name'])

    elif 'deeppavlov' in system.name:
        data = {'context': [test_sentence.text]}
        r = requests.post('http://localhost:{}/intents'.format(get_port(system.name)), data=json.dumps(data),
                          headers=Header.json.value)
        return IntentClassification(system, r.json()[0][0][0])

    elif 'watson' in system:
        raise AssertionError('Not implemented. Possibly interesting: https://github.com/joe4k/wdcutils/')
