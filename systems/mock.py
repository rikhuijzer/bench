from systems.systems import System, Corpus


def train(system: System, corpus: Corpus) -> System:
    data = list(system.data)
    data[0] += 1
    return System(system.name, corpus, tuple(data))
