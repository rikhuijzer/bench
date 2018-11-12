import functools

import yaml

import src.typ as tp
import src.utils
import src.systems.amazon_lex
import src.systems.deeppavlov
import src.systems.mock
import src.systems.rasa
import src.systems.watson
import src.systems.dialogflow

import logging
from typing import Callable


@functools.lru_cache(maxsize=1)
def get_docker_compose_configuration() -> dict:
    with open(str(src.utils.get_root() / 'docker-compose.yml'), 'rb') as f:
        return yaml.load(f)


def get_n_systems() -> int:
    return len(get_docker_compose_configuration()['services'])


def get_port(system: str) -> int:
    return int(get_docker_compose_configuration()['services'][system]['ports'][0][0:4])


def get_header(header: tp.Header) -> dict:
    if header == tp.Header.JSON:
        return {'content-type': 'application/json'}
    if header == tp.Header.YML:
        return {'content-type': 'application/x-yml'}


def update_timestamp(system: tp.System) -> tp.System:
    # return tp.System(system.name, system.knowledge, src.utils.get_timestamp(), system.data)
    return system._replace(timestamp=src.utils.get_timestamp())


def add_retrain(system: tp.System) -> tp.System:
    """Add retrain flag."""
    return system._replace(data=system.data + ('retrain', ))


def remove_retrain(system: tp.System) -> tp.System:
    """Remove retrain flag."""
    # return tp.System(system.name, system.knowledge, tuple(filter(lambda x: x != 'retrain', system.data)))
    return system._replace(data=tuple(filter(lambda x: x != 'retrain', system.data)))


def train(system_corpus: tp.SystemCorpus) -> tp.System:
    """ Train system on corpus. """
    logging.info('Training {} on {}...'.format(system_corpus.system, system_corpus.corpus))

    system = remove_retrain(system_corpus.system)
    system = update_timestamp(system)

    train_systems = {
        'mock': src.systems.mock.train,
        'rasa': src.systems.rasa.train,
        'deeppavlov': src.systems.deeppavlov.train,
        'lex': src.systems.amazon_lex.train,
        'dialogflow': src.systems.dialogflow.train,
    }
    func: Callable[[tp.SystemCorpus], tp.System] = src.utils.get_substring_match(train_systems, system.name)
    return func(system_corpus._replace(system=system))


def get_classification(system: tp.System, message: tp.Message) -> tp.Classification:
    """ Get intent for some system and some sentence. Function will train system if that is necessary. """
    if message.data['corpus'] != system.knowledge or 'retrain' in system.data:
        system = train(tp.SystemCorpus(system, message.data['corpus']))
    elif not system.timestamp:
        system = update_timestamp(system)

    get_intent_systems = {
        'mock': src.systems.mock.get_response,
        'rasa': src.systems.rasa.get_response,
        'watson': src.systems.watson.get_response,
        'amazon': src.systems.amazon_lex.get_response,
    }

    func: Callable[[tp.Query], tp.Response] = src.utils.get_substring_match(get_intent_systems, system.name)
    query = tp.Query(system, message.text)
    response = func(query)
    return tp.Classification(system, message, response)
