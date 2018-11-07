import functools

import yaml

import src.typ
import src.utils
import src.systems.amazon_lex
import src.systems.deeppavlov
import src.mock
import src.systems.rasa
import src.systems.watson

import logging


@functools.lru_cache(maxsize=1)
def get_docker_compose_configuration() -> dict:
    with open(str(src.utils.get_root() / 'docker-compose.yml'), 'rb') as f:
        return yaml.load(f)


def get_n_systems() -> int:
    return len(get_docker_compose_configuration()['services'])


def get_port(system: str) -> int:
    return int(get_docker_compose_configuration()['services'][system]['ports'][0][0:4])


def get_header(header: src.typ.Header) -> dict:
    if header == src.typ.Header.JSON:
        return {'content-type': 'application/json'}
    if header == src.typ.Header.YML:
        return {'content-type': 'application/x-yml'}


def train(sc: src.typ.SystemCorpus) -> src.typ.System:
    """ Train system on corpus. """
    logging.info('Training {} on {}...'.format(sc.system, sc.corpus))

    train_systems = {  # src.typ.SystemCorpus -> src.typ.System
        'mock': src.mock.train,
        'rasa': src.systems.rasa.train,
        'deeppavlov': src.systems.deeppavlov.train,
        'lex': src.systems.amazon_lex.train
    }

    func = src.utils.get_substring_match(train_systems, sc.system.name)
    return func(sc)


def get_classification(system: src.typ.System, message: src.typ.Message) -> src.typ.Classification:
    """ Get intent for some system and some sentence. Function will train system if that is necessary. """
    if message.data['corpus'] != system.knowledge or 'retrain' in system.data:
        system = src.typ.System(system.name, system.knowledge, tuple(filter(lambda x: x != 'retrain', system.data)))
        system = train(src.typ.SystemCorpus(system, message.data['corpus']))

    get_intent_systems = {  # src.typ.Query -> src.typ.Response
        'mock': src.mock.get_response,
        'rasa': src.systems.rasa.get_response,
        'watson': src.systems.watson.get_response,
        'amazon': src.systems.amazon_lex.get_response,
    }

    func = src.utils.get_substring_match(get_intent_systems, system.name)
    query = src.typ.Query(system, message.text)
    response = func(query)
    return src.typ.Classification(system, message, response)
