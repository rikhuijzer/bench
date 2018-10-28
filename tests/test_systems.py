from systems.systems import *


def test_get_docker_compose_configuration():
    assert 'version' in get_docker_compose_configuration()


def test_get_n_systems():
    assert 3 < get_n_systems()


def test_get_port():
    services = get_docker_compose_configuration()['services']
    for system in services:
        assert 5000 <= get_port(system) < 6000


def test_train():
    assert System('mock', Corpus.WebApplications) == train('mock', Corpus.WebApplications)


def test_get_intent():
    system = System('mock', Corpus.Empty)
    assert IntentClassification(system, 'A') == get_intent(system, TestSentence('2', Corpus.Empty))

    trained_system = System('mock', Corpus.Chatbot)
    assert IntentClassification(trained_system, 'A') == get_intent(system, TestSentence('2', Corpus.Chatbot))
