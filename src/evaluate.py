import src.typ
import src.training_data
import typing
import rasa_nlu.training_data
import src.results


def classify(sc: src.typ.SystemCorpus, m: rasa_nlu.training_data.Message) -> src.typ.Classification:
    """ Transform a Rasa Message to a Classification. """
    return src.system.get_classification(sc.system, src.typ.Sentence(m.text, sc.corpus))


def get_classifications(sc: src.typ.SystemCorpus) -> typing.Iterable[src.typ.Classification]:
    if not isinstance(sc, src.typ.SystemCorpus):
        raise AssertionError('incorrect parameter type')

    """ Run all test sentences from some corpus through system and return classifications. """
    messages = src.training_data.get_filtered_messages(sc.corpus, train=False)

    for message in messages:
        # It seems difficult to do this by map, filter, reduce since the system state changes.
        classification = classify(sc, message)
        sc = classification.system_corpus
        yield classification


def run_bench(sc: src.typ.SystemCorpus):
    classifications = get_classifications(sc)

    for classification in classifications:
        src.results.write_classification(classification)
