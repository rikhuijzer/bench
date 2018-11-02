import core.typ


def train(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.System:
    data = list(system.data)
    data[0] += 1
    return core.typ.System(system.name, corpus, tuple(data))


def get_intent(system: core.typ.System, test_sentence: core.typ.TestSentence) -> core.typ.IntentClassification:
    value = int(test_sentence.text)
    classifications = 'A' if (0 < value <= 10) else 'B'
    return core.typ.IntentClassification(system, 'C' if (value % system.data[0] == 0) else classifications)
