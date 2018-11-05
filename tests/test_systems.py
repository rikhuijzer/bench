import core.typ
import systems.systems

# Only testing logic and not specific system logic to have speedy tests and avoiding many API calls.


def test_get_docker_compose_configuration():
    assert 'version' in systems.systems.get_docker_compose_configuration()


def test_get_n_systems():
    assert 3 < systems.systems.get_n_systems()


def test_get_port():
    services = systems.systems.get_docker_compose_configuration()['services']
    for system in services:
        assert 5000 <= systems.systems.get_port(system) < 6000


def test_train():
    expected = core.typ.System('mock', core.typ.Corpus.WebApplications, (3, ))
    system = core.typ.System('mock', core.typ.Corpus.Empty, (2, ))
    corpus = core.typ.Corpus.WebApplications
    assert expected == systems.systems.train(core.typ.SystemCorpus(system, corpus))


def test_get_classification():
    """ In the tuple we define the modulus to be used. """
    untrained_system = core.typ.System('mock', core.typ.Corpus.Empty, (3, ))
    print('foo: ', untrained_system)



    corpus = core.typ.Corpus.Mock
    untrained_sc = core.typ.SystemCorpus(untrained_system, corpus)
    print('bar: ', untrained_system)
    classification = systems.systems.get_classification(untrained_system, core.typ.Sentence('2', corpus))
    assert core.typ.Classification(untrained_sc, core.typ.Response('A', 1.0, [])) == classification

    """ During training the modulus is increased by one to model the changing behaviour of a probabilistic system. """
    system = classification.system_corpus.system
    classification = systems.systems.get_classification(system, core.typ.Sentence('2', corpus))
    trained_system = core.typ.System('mock', corpus, (4, ))
    assert trained_system == classification.system_corpus.system
    assert core.typ.Response('A', 1.0, []) == classification.response
