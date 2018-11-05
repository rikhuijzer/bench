import functools

import yaml

import core.typ
import core.utils
import systems.amazon_lex
import systems.deeppavlov
import systems.mock
import systems.rasa
import systems.watson
import typing


@functools.lru_cache(maxsize=1)
def get_docker_compose_configuration() -> dict:
    with open(str(core.utils.get_root() / 'docker-compose.yml'), 'rb') as f:
        return yaml.load(f)


def get_n_systems() -> int:
    return len(get_docker_compose_configuration()['services'])


def get_port(system: str) -> int:
    return int(get_docker_compose_configuration()['services'][system]['ports'][0][0:4])


def train(sc: core.typ.SystemCorpus) -> core.typ.System:
    """ Train system on corpus. """
    if sc.system.knowledge != core.typ.Corpus.Mock:
        print('Training {} on {}...'.format(sc.system, sc.corpus))

    train_systems = {  # core.typ.SystemCorpus -> core.typ.System
        'mock': systems.mock.train,
        'rasa': systems.rasa.train,
        'deeppavlov': systems.deeppavlov.train,
        'lex': systems.amazon_lex.train
    }

    func = core.utils.get_substring_match(train_systems, sc.system.name)
    return func(sc)


def get_classification(system: core.typ.System, test_sentence: core.typ.Sentence) -> core.typ.Classification:
    """ Get intent for some system and some sentence. Function will train system if that is necessary. """
    if test_sentence.corpus != system.knowledge or 'retrain' in system.data:
        system = core.typ.System(system.name, system.knowledge, tuple(filter(lambda x: x != 'retrain', system.data)))
        system = train(core.typ.SystemCorpus(system, test_sentence.corpus))

    get_intent_systems = {  # core.typ.Query -> core.typ.Response
        'mock': systems.mock.get_response,
        'rasa': systems.rasa.get_response,
        'watson': systems.watson.get_response,
        'amazon': systems.amazon_lex.get_response,
    }

    func = core.utils.get_substring_match(get_intent_systems, system.name)
    query = core.typ.Query(system, test_sentence.text)
    response = func(query)
    return core.typ.Classification(system, response)
