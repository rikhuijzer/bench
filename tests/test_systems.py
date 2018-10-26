from systems.systems import *


def test_docker_compose_configuration():
    assert 'version' in get_docker_compose_configuration()


def test_n_systems():
    assert get_n_systems() > 3


def test_port():
    services = get_docker_compose_configuration()['services']
    for system in services:
        assert 5000 <= get_port(system) < 6000
        

def test_train():
    assert train('rasa-spacy', Corpus.WebApplications)
