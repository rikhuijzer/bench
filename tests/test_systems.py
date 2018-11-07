import src.typ
import src.system
import src.training_data

# Only testing logic and not specific system logic to have speedy tests and avoiding many API calls.


def test_get_docker_compose_configuration():
    assert 'version' in src.system.get_docker_compose_configuration()


def test_get_n_systems():
    assert 3 < src.system.get_n_systems()


def test_get_port():
    services = src.system.get_docker_compose_configuration()['services']
    for system in services:
        assert 5000 <= src.system.get_port(system) < 6000


def test_train():
    expected = src.typ.System('mock', src.typ.Corpus.WEBAPPLICATIONS, (3, ))
    system = src.typ.System('mock', src.typ.Corpus.EMPTY, (2, ))
    corpus = src.typ.Corpus.WEBAPPLICATIONS
    assert expected == src.system.train(src.typ.SystemCorpus(system, corpus))


def test_get_classification():
    """ In the tuple we define the modulus to be used. """
    untrained_system = src.typ.System('mock', src.typ.Corpus.EMPTY, (3, ))
    corpus = src.typ.Corpus.MOCK
    message = src.training_data.create_message('2', '', [], False, corpus)
    classification = src.system.get_classification(untrained_system, message)
    assert src.typ.Response('A', 1.0, []) == classification.response

    """ During training the modulus is increased by one to model the changing behaviour of a probabilistic system. """
    system = classification.system
    classification = src.system.get_classification(system, message)
    trained_system = src.typ.System('mock', corpus, (4, ))
    assert trained_system == classification.system
    assert src.typ.Response('A', 1.0, []) == classification.response
