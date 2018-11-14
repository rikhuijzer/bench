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
import logging


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
    newest_tuple = src.results.get_newest_tuple(system_corpus, csv)
    if not newest_tuple:
        raise AssertionError('No newest tuple exists. Trying to evaluate without running benchmark first? Occured on:\n'
                             '{} and {}'.format(system_corpus, csv))
    return filter(lambda x: x.timestamp == newest_tuple.timestamp, src.results.get_elements(system_corpus, csv))


def get_f1_intent(system_corpus: tp.SystemCorpus, average='micro') -> float:
    """Returns f1 score for last run of some system and corpus."""
    csv_intents = tuple(get_previous_run(system_corpus, csv=tp.CSVs.INTENTS))

    y_true = tuple(map(lambda e: e.gold_standard, csv_intents))
    y_pred = tuple(map(lambda e: e.classification, csv_intents))
    return round(f1_score(y_true, y_pred, average=average), 3)


def get_statistics(system_corpus: tp.SystemCorpus) -> dict:
    """Returns dict which can be converted to yml to be put into statistics.yml."""
    f1_intent = partial(get_f1_intent, system_corpus=system_corpus)
    averages = ['micro', 'macro', 'weighted']

    f1_scores = {
        'micro': f1_intent('micro'),
        'macro': f1_intent('macro'),
        'weighted': f1_intent('weighted')
    }
    return {
        'system name': system_corpus.system.name,
        'corpus': system_corpus.corpus,
        'f1 scores': f1_scores
    }


def write_statistics(system_corpus: tp.SystemCorpus) -> bool:
    """Write statistics for last run to statistics.yml."""
    stats = get_statistics(system_corpus)
    logging.info('Writing the following statistics to file:\n {}'.format(stats))


    return True
