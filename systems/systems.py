import functools

import yaml

import core.typ
import core.utils
import systems.amazon_lex
import systems.deeppavlov
import systems.mock
import systems.rasa
import systems.watson


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

    fn = core.utils.get_substring_match(train_systems, system.name)
    return fn(system, corpus)


def get_intent(system: core.typ.System, test_sentence: core.typ.TestSentence) -> core.typ.IntentClassification:
    """ Get intent for some system and some sentence. Function will train system if that is necessary. """
    if test_sentence.corpus != system.knowledge or 'retrain' in system.data:
        system = core.typ.System(system.name, system.knowledge, tuple(filter(lambda x: x != 'retrain', system.data)))
        system = train(system, test_sentence.corpus)

    get_intent_systems = {
        'mock': systems.mock.get_response,
        'rasa': systems.rasa.get_response,
        'watson': systems.watson.get_response,
        'amazon': systems.amazon_lex.get_response,
    }

    fn = core.utils.get_substring_match(get_intent_systems, system.name)
    return fn(system, test_sentence)
