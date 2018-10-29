from systems.systems import *
# Only testing logic and not specific system logic to have speedy tests and avoiding many API calls.


def test_get_docker_compose_configuration():
    assert 'version' in get_docker_compose_configuration()


def test_get_n_systems():
    assert 3 < get_n_systems()


def test_get_port():
    services = get_docker_compose_configuration()['services']
    for system in services:
        assert 5000 <= get_port(system) < 6000


def test_train():
    assert System('mock', Corpus.WebApplications, (3, )) == train(System('mock', Corpus.Empty, (2, )),
                                                                  Corpus.WebApplications)


def test_get_intent():
    """ In the tuple we define the modulus to be used. """
    system = System('mock', Corpus.Empty, (3, ))
    assert IntentClassification(system, 'A') == get_intent(system, TestSentence('2', Corpus.Empty))

    """ During training the modulus is increased by one to model the changing behaviour of a probabilistic system. """
    trained_system = System('mock', Corpus.Chatbot, (4, ))
    assert IntentClassification(trained_system, 'A') == get_intent(system, TestSentence('2', Corpus.Chatbot))
