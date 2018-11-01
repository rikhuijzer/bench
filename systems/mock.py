import core.typ


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    data = list(system.data)
    data[0] += 1
    return core.typ.System(system.name, corpus, tuple(data))
