import core.typ


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    data = list(system.data)
    data[0] += 1
    return core.typ.System(system.name, corpus, tuple(data))


def get_response(query: core.typ.Query) -> core.typ.Response:
    value = int(query.text)
    classification = 'C' if (value % query.system.data[0] == 0) else ('A' if (0 < value <= 10) else 'B')
    return core.typ.Response(classification, 1.0, [])
