from systems.systems import *


def test_get_docker_compose_configuration():
    assert 'version' in get_docker_compose_configuration()


def test_get_n_systems():
    assert 3 < get_n_systems()


def test_get_port():
    services = get_docker_compose_configuration()['services']
    for system in services:
        assert 5000 <= get_port(system) < 6000


def test_get_intent():
    assert 'Find Alternative' == get_intent('rasa-spacy', 'Alternatives to Twitter', Corpus.WebApplications, None)
    assert 'PlayMusic' == get_intent('deeppavlov-snips', '', None, None)
