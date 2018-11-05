import core.typ
import sklearn.metrics
import core.training_data
import typing
import rasa_nlu.training_data
import systems.systems
import core.results


def classify(sc: core.typ.SystemCorpus, m: rasa_nlu.training_data.Message) -> core.typ.Classification:
    """ Transform a Rasa Message to a Classification. """
    return systems.systems.get_classification(sc.system, core.typ.Sentence(m.text, sc.corpus))


def get_classifications(sc: core.typ.SystemCorpus) -> typing.Iterable[core.typ.Classification]:
    if not isinstance(sc, core.typ.SystemCorpus):
        raise AssertionError('incorrect parameter type')

    """ Run all test sentences from some corpus through system and return classifications. """
    messages = core.training_data.get_filtered_messages(sc.corpus, train=False)

    for message in messages:
        # It seems difficult to do this by map, filter, reduce since the system state changes.
        classification = classify(sc, message)
        sc = classification.system_corpus
        yield classification


def run_bench(sc: core.typ.SystemCorpus):
    classifications = get_classifications(sc)

    for classification in classifications:
        core.results.write_classification(classification)
