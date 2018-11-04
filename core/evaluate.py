import core.typ
import sklearn.metrics
import core.training_data
import typing
import rasa_nlu.training_data
import systems.systems


def get_classifications(system: core.typ.System, corpus: core.typ.Corpus) -> typing.Iterable[core.typ.Classification]:
    """ Run all test sentences from some corpus through system and return results. """
    messages = core.training_data.get_filtered_messages(corpus, train=False)

    def classify(s: core.typ.System, m: rasa_nlu.training_data.Message) -> core.typ.Classification:
        """ Transform a Rasa Message to a Classification. Note that corpus is defined in outer function. """
        return systems.systems.get_classification(s, core.typ.Sentence(m.text, corpus))

    for message in messages:
        # It is difficult to replace this by map, filter, reduce since the system state can change.
        classification = classify(system, message)
        system = classification.system
        yield classification
