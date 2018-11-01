import core.typ


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    raise AssertionError('Trying to train {} which should be trained via Docker build'.format(system))
