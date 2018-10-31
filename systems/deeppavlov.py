from systems.systems import System, Corpus


def train(system: System, corpus: Corpus) -> System:
    raise AssertionError('Trying to train {} which should be trained via Docker build'.format(system))
