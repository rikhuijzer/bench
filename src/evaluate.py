import logging
import os
import typing
from pathlib import Path
from typing import Iterable, Optional

import oyaml as yaml
import rasa_nlu.training_data
from sklearn.metrics import f1_score

import src.dataset
import src.results
import src.system
import src.typ as tp
from src.system import add_retrain
from src.utils import get_root, iterate


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


def write_classifications(system_corpus: tp.SystemCorpus, n_runs=1) -> Iterable[tp.Classification]:
    for _ in range(0, n_runs):
        classifications = get_classifications(system_corpus, retrain=True)

        for classification in classifications:
            src.results.write_classification(classification)
            yield classification


def get_previous_run(system_corpus: tp.SystemCorpus, csv: tp.CSVs) -> Iterable[tp.CSV_types]:
    """Get all classifications for runs with most recent timestamp."""
    newest_tuple = src.results.get_newest_tuple(system_corpus, csv)
    if not newest_tuple:
        raise AssertionError('No newest tuple exists. Occured on:\n {} and {}'.format(system_corpus, csv))
    return filter(lambda x: x.timestamp == newest_tuple.timestamp, src.results.get_elements(system_corpus, csv))


def get_f1_intent(system_corpus: tp.SystemCorpus, average='micro') -> float:
    """Returns f1 score for last run of some system and corpus."""
    csv_intents = tuple(get_previous_run(system_corpus, csv=tp.CSVs.INTENTS))

    y_true = tuple(map(lambda e: e.gold_standard, csv_intents))
    y_pred = tuple(map(lambda e: e.classification, csv_intents))
    return round(f1_score(y_true, y_pred, average=average), 3)


def get_statistics(system_corpus: tp.SystemCorpus) -> dict:
    """Returns dict which can be converted to yml to be put into statistics.yml."""
    averages = ['micro', 'macro', 'weighted']
    f1_scores = tuple(map(lambda average: str(get_f1_intent(system_corpus, average)), averages))
    return {
        'system name': system_corpus.system.name,
        'corpus': str(system_corpus.corpus),
        'f1 intent scores': dict(zip(averages, f1_scores))
    }


def write_statistics(system_corpus: tp.SystemCorpus) -> bool:
    """Write statistics for last run to statistics.yml."""
    stats = get_statistics(system_corpus)
    logging.info('Writing the following statistics:\n {}'.format(stats))

    filename = src.results.get_filename(system_corpus, tp.CSVs.STATS)
    with open(str(filename), 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    return True


def get_summary_filename() -> Path:
    return get_root() / 'results' / 'summary.yml'


def read_yaml(filename: Path) -> Optional[dict]:
    """Returns yaml file content as dict if file exists."""
    if not os.path.isfile(filename):
        return None
    else:
        with open(str(filename), 'r') as f:
            return yaml.load(f)


def read_summary() -> dict:
    """Returns summary as specified in file or empty dict if file does not exists."""
    filename = get_summary_filename()
    content = read_yaml(filename)
    return content if content else {}


def write_summary(summary: dict):
    """Dump summary to summary.yml."""
    with open(str(get_summary_filename()), 'w') as f:
        return yaml.dump(summary, f, default_flow_style=False)


def add_statistic(filename: Path, summary: dict) -> dict:
    """Returns updated summary using statistic information from some file."""
    content = read_yaml(filename)
    corpus = content['corpus']
    if corpus != 'Corpus.MOCK':
        system_name = content['system name']

        averages = ['micro', 'macro', 'weighted']
        scores = map(lambda average: content['f1 intent scores'][average], averages)
        for average, score in zip(averages, scores):
            src.utils.add_nested_value(summary, score, corpus, 'f1 intent scores', average, system_name)
    return summary


def add_statistics(summary: dict) -> dict:
    results_folder = get_root() / 'results'
    folders = filter(lambda f: f.is_dir(), results_folder.glob('./*'))
    for folder in folders:
        filename = results_folder / folder / 'statistics.yml'
        if os.path.isfile(filename):
            summary = add_statistic(filename, summary)
    return summary


def update_summary():
    """Reads statistics from all sub-folders of 'results' and summarizes them."""
    summary = read_summary()
    summary = add_statistics(summary)
    write_summary(summary)


def evaluate(system_corpus: tp.SystemCorpus) -> bool:
    iterate(write_classifications(system_corpus))
    write_statistics(system_corpus)
    update_summary()
    return True
