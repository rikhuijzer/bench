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
    assert core.typ.System('mock', core.typ.Corpus.WebApplications, (3, )) == \
           systems.systems.train(core.typ.System('mock', core.typ.Corpus.Empty, (2, )), core.typ.Corpus.WebApplications)


def test_get_intent():
    """ In the tuple we define the modulus to be used. """
    system = core.typ.System('mock', core.typ.Corpus.Empty, (3, ))
    classification = systems.systems.get_intent(system, core.typ.TestSentence('2', core.typ.Corpus.Empty))
    assert core.typ.Classification(system, core.typ.Response('A', 1.0, [])) == classification

    """ During training the modulus is increased by one to model the changing behaviour of a probabilistic system. """
    trained_system = core.typ.System('mock', core.typ.Corpus.Chatbot, (4, ))
    classification = systems.systems.get_intent(system, core.typ.TestSentence('2', core.typ.Corpus.Chatbot))
    assert core.typ.Classification(trained_system, core.typ.Response('A', 1.0, [])) == classification
