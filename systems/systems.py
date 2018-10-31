import json
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import requests
import yaml

from core.training_data import Corpus, TestSentence
from systems.amazon_lex import get_intent_lex
import systems.mock
import systems.rasa
import systems.deeppavlov


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

    train_systems = {
        'mock': systems.mock.train,
        'rasa': systems.rasa.train,
        'deeppavlov': systems.deeppavlov.train,
        'lex': systems.amazon_lex.train
    }

    match = tuple(filter(lambda key: system.name in key, train_systems))
    if len(match) == 0:
        raise ValueError('Unknown system for training: {}.'.format(system.name))
    if len(match) > 1:
        AssertionError('Unable to train {} since multiple systems contain {}.'.format(system, system.name))

    func_train = train_systems[match[0]]
    return func_train(system, corpus)


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

    if 'watson' in system.name:
        from watson_developer_cloud import AssistantV1
        import time
        time.sleep(1)
        default_url = 'https://gateway.watsonplatform.net/assistant/api'  # might differ based on workspace
        assistant = AssistantV1(version='2018-09-20', username=os.environ['WATSON_USERNAME'],
                                password=os.environ['WATSON_PASSWORD'], url=default_url)
        response = assistant.message(workspace_id='c6548076-8034-4f28-a155-ab546b0058d5',
                                     input={'text': test_sentence.text},
                                     alternate_intents=False).get_result()
        print(response['intents'])
        if response['intents']:
            classification = response['intents'][0]['intent'].replace('_', ' ')
        else:
            classification = ''
        return IntentClassification(system, classification)

    if 'deeppavlov' in system.name:
        data = {'context': [test_sentence.text]}
        r = requests.post('http://localhost:{}/intents'.format(get_port(system.name)), data=json.dumps(data),
                          headers=Header.json.value)
        return IntentClassification(system, r.json()[0][0][0])

    if 'watson' in system:
        raise AssertionError('Not implemented. Possibly interesting: https://github.com/joe4k/wdcutils/')

    if 'amazon' in system:
        return get_intent_lex()

    raise AssertionError('Unknown system.name for {}'.format(system))
