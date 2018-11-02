import core.typ


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    data = list(system.data)
    data[0] += 1
    return core.typ.System(system.name, corpus, tuple(data))


def get_response(system: core.typ.System, test_sentence: core.typ.TestSentence) -> core.typ.Classification:
    value = int(test_sentence.text)
    classification = 'C' if (value % system.data[0] == 0) else ('A' if (0 < value <= 10) else 'B')
    return core.typ.Classification(classification, '1.0', [])
