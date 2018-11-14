import src.typ as tp
import src.dataset
import typing
import rasa_nlu.training_data
import src.results
import src.system
from src.system import add_retrain
from sklearn.metrics import f1_score
from typing import Iterable, Callable
from functools import partial


def classify(system: tp.System, message: rasa_nlu.training_data.Message) -> tp.Classification:
    """ Transform a Rasa Message to a Classification. """
    return src.system.get_classification(system, message)


def get_classifications(system_corpus: tp.SystemCorpus, retrain: bool) -> typing.Iterable[tp.Classification]:
    if not isinstance(system_corpus, tp.SystemCorpus):
        raise AssertionError('incorrect parameter type')

    """ Run all test sentences from some corpus through system and return classifications. """
    messages = src.dataset.get_filtered_messages(system_corpus.corpus, train=False)

    system = add_retrain(system_corpus.system) if retrain else system_corpus.system
    for message in messages:
        # It seems difficult to do this by map, filter, reduce since the system state changes.
        classification = classify(system, message)
        system = classification.system
        yield classification


def run_bench(system_corpus: tp.SystemCorpus, n_runs=1) -> Iterable[tp.Classification]:
    for _ in range(0, n_runs):
        classifications = get_classifications(system_corpus, retrain=True)

        for classification in classifications:
            src.results.write_classification(classification)
            yield classification


def get_previous_run(system_corpus: tp.SystemCorpus, csv: tp.CSVs) -> Iterable[tp.CSV_types]:
    """Get all classifications for runs with most recent timestamp. Will crash if there is no previous run."""
    timestamp = src.results.get_newest_tuple(system_corpus, csv).timestamp
    return filter(lambda x: x.timestamp == timestamp, src.results.get_elements(system_corpus, csv))


def get_f1_intent(system_corpus: tp.SystemCorpus, run: tp.Run, average='micro') -> float:
    """Returns f1 score for some system and corpus. Does this for ONE new run or one or more old runs."""
    def get_new_intents(sc: tp.SystemCorpus) -> Iterable[tp.CSVIntent]:
        return map(lambda x: src.results.get_csv_intent(x), run_bench(sc))

    functions = {
        tp.Run.PREV: partial(get_previous_run, csv=tp.CSVs.INTENTS),
        tp.Run.ALL: partial(src.results.get_elements, csv=tp.CSVs.INTENTS),
        tp.Run.NEW: get_new_intents
    }
    func: Callable[[tp.SystemCorpus], Iterable[tp.CSVIntent]] = functions[run]
    elements = tuple(func(system_corpus))

    y_true = tuple(map(lambda e: e.intent, elements))
    y_pred = tuple(map(lambda e: e.classification, elements))
    return round(f1_score(y_true, y_pred, average=average), 3)
