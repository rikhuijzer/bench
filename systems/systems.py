import json
import os
import functools

import requests
import yaml
import core.typ
import systems.deeppavlov
import systems.amazon_lex
import systems.mock
import systems.rasa
import core.utils


@functools.lru_cache(maxsize=1)
def get_docker_compose_configuration() -> dict:
    with open(str(core.utils.get_root() / 'docker-compose.yml'), 'rb') as f:
        return yaml.load(f)


def get_n_systems() -> int:
    return len(get_docker_compose_configuration()['services'])


def get_port(system: str) -> int:
    return int(get_docker_compose_configuration()['services'][system]['ports'][0][0:4])


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    """ Train system on corpus. """
    if system.knowledge != core.typ.Corpus.Mock:
        print('Training {} on {}...'.format(system, corpus))

    train_systems = {
        'mock': systems.mock.train,
        'rasa': systems.rasa.train,
        'deeppavlov': systems.deeppavlov.train,
        'lex': systems.amazon_lex.train
    }

    match = tuple(filter(lambda key: key in system.name, train_systems))
    if len(match) == 0:
        raise ValueError('Unknown system for training: {}.'.format(system.name))
    if len(match) > 1:
        AssertionError('Unable to train {} since multiple systems contain {}.'.format(system, system.name))

    func_train = train_systems[match[0]]
    return func_train(system, corpus)


def get_intent(system: core.typ.System, test_sentence: core.typ.TestSentence) -> core.typ.IntentClassification:
    """ Get intent for some system and some sentence. Function will train system if that is necessary. """
    if test_sentence.corpus != system.knowledge or 'retrain' in system.data:
        system = core.typ.System(system.name, system.knowledge, tuple(filter(lambda x: x != 'retrain', system.data)))
        system = train(system, test_sentence.corpus)

    if 'mock' == system.name:
        value = int(test_sentence.text)
        classifications = 'A' if (0 < value <= 10) else 'B'
        return core.typ.IntentClassification(system, 'C' if (value % system.data[0] == 0) else classifications)

    if 'rasa' in system.name:
        data = {'q': test_sentence.text, 'project': 'my_project'}
        url = 'http://localhost:{}/parse'
        r = requests.post(url.format(get_port(system.name)), data=json.dumps(data), headers=core.typ.Header.json.value)
        print(r.json())
        if r.status_code != 200:
            raise RuntimeError('Could not get intent for text: {}'.format(test_sentence.text))
        return core.typ.IntentClassification(system, r.json()['intent']['name'])

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
        return core.typ.IntentClassification(system, classification)

    if 'deeppavlov' in system.name:
        data = {'context': [test_sentence.text]}
        r = requests.post('http://localhost:{}/intents'.format(get_port(system.name)), data=json.dumps(data),
                          headers=core.typ.Header.json.value)
        return core.typ.IntentClassification(system, r.json()[0][0][0])

    if 'watson' in system:
        raise AssertionError('Not implemented. Possibly interesting: https://github.com/joe4k/wdcutils/')

    if 'amazon' in system:
        return systems.amazon_lex.get_intent_lex()

    raise AssertionError('Unknown system.name for {}'.format(system))
