import functools
import logging
import os
import pathlib
from typing import Iterable, Tuple, NamedTuple
from rasa_nlu.training_data.message import Message

import src.evaluate
import src.typ as tp
import src.utils
from typing import Optional


def get_folder(sc: tp.SystemCorpus) -> pathlib.Path:
    return src.utils.get_root() / 'results' / '{}-{}'.format(sc.system.name, sc.corpus.name)


def get_filename(sc: tp.SystemCorpus, csv: tp.CSVs) -> pathlib.Path:
    """ Returns the filename for some system and corpus and CSV type. """
    mapping = {
        tp.CSVs.GENERAL: 'general.yml',
        tp.CSVs.INTENTS: 'intents.csv',
        tp.CSVs.ENTITIES: 'entities.csv'
    }
    return get_folder(sc) / mapping[csv]


def append_text(text: str, filename: pathlib.Path):
    """ Append text to a file, this function assumes the file exists. """
    with open(str(filename), 'a') as f:
        f.write(text + '\n')


def convert_tuple_str(t: Tuple) -> str:
    """Converting tuple to string. Removing comma's to avoid problems when reading. Good enough solution for now."""
    return ','.join([str(value).replace(',', '') for value in t])


def get_tuple_types(t: type) -> Iterable[type]:
    """ Get all the types of the fields in some NamedTuple. """
    return map(lambda x: x[1], t._field_types.items())


def get_csv_type(csv: tp.CSVs) -> type:
    """ Get the corresponding type of NamedTuple for some CSV type. """
    mapping = {
        tp.CSVs.GENERAL: tp.CSVGeneral,
        tp.CSVs.INTENTS: tp.CSVIntent,
        tp.CSVs.ENTITIES: tp.CSVEntity
    }
    return mapping[csv]


def split_on_comma(text: str) -> Iterable[str]:
    out = []
    import csv
    return out

def convert_str_tuple(text: str, csv: tp.CSVs) -> tp.CSV_types:
    """ Convert a string from csv back to NamedTuple. """
    tuple_types = get_tuple_types(get_csv_type(csv))
    converted = map(lambda t: t[0](t[1]), zip(tuple_types, text.split(',')))

    if csv == tp.CSVs.INTENTS:
        return tp.CSVIntent(*converted)
    else:
        raise AssertionError('not implemented')


@functools.lru_cache()
def create_folder(path: pathlib) -> bool:
    """ Creates folder. Works even if folder already exists. Is cached to reduce filesystem operations. """
    path.mkdir(parents=False, exist_ok=True)
    return True


@functools.lru_cache()
def create_file(path: pathlib, header: str) -> bool:
    """ Creates file and adds header if file does not already exists. Is cached to reduce filesystem operations. """
    if not os.path.isfile(path):
        with open(str(path), 'a+') as f:
            f.write(header + '\n')
    return True


def create_header(csv: tp.CSVs) -> str:
    """ Create a header which can be placed in first line of CSV. """
    return ','.join([name for name in csv.__annotations__])


@functools.lru_cache()
def initialize_file(sc: tp.SystemCorpus, csv: tp.CSVs) -> pathlib.Path:
    create_folder(get_folder(sc))

    if csv == tp.CSVIntent:
        filename = get_filename(sc, tp.CSVs.INTENTS)
        create_file(filename, create_header(csv))
    else:
        # TODO: Accept other NamedTuples
        raise AssertionError('src.write.write_tuple got invalid input t: {}'.format(namedtuple))
    return filename


def write_tuple(sc: tp.SystemCorpus, namedtuple: NamedTuple):
    """ Write some tuple to CSV. Also creates folder and file if file does not yet exist. """
    logging.info('Writing {}'.format(namedtuple))

    filename = initialize_file(sc, type(namedtuple))
    append_text(convert_tuple_str(namedtuple), filename)


def read_file(sc: tp.SystemCorpus, csv: tp.CSVs) -> str:
    filename = get_filename(sc, csv)

    if not os.path.isfile(filename):
        raise AssertionError('Tried to read {}. It seems that this file does not exist.'.format(filename))

    with open(str(filename), 'r') as f:
        return f.read()


def get_newest_tuple(sc: tp.SystemCorpus, csv: tp.CSVs) -> Optional[tp.CSV_types]:
    """ Get the id for the newest tuple (last line) in the csv. """
    if not os.path.isfile(get_filename(sc, csv)):
        return None

    file_content = read_file(sc, csv)
    file_content = file_content.strip()
    last_line = file_content[file_content.rfind('\n'):]
    return convert_str_tuple(last_line, csv)


def get_csv_intent(classification: tp.Classification) -> tp.CSVIntent:
    """ Convert classification to intent tuple which can be sent to CSV. """
    system_corpus = tp.SystemCorpus(classification.system, classification.message.data['corpus'])
    newest_tuple = get_newest_tuple(system_corpus, tp.CSVs.INTENTS)
    return tp.CSVIntent(id=newest_tuple.id + 1 if newest_tuple else 0,
                        timestamp=classification.system.timestamp,
                        sentence=classification.message.text,
                        intent=classification.message.data['intent'],
                        classification=classification.response.intent,
                        confidence=classification.response.confidence,
                        time=0)


def write_classification(classification: tp.Classification):
    """Unpack and write a classification to various files."""
    csv_intent = get_csv_intent(classification)
    system_corpus = tp.SystemCorpus(classification.system, classification.message.data['corpus'])
    write_tuple(system_corpus, csv_intent)


def get_elements(sc: tp.SystemCorpus, csv: tp.CSVs) -> Iterable[tp.CSV_types]:
    """Get all elements from some results CSV file."""
    content = read_file(sc, csv)
    lines = content.strip().split('\n')

    return map(lambda line: convert_str_tuple(line, csv), lines[1:])
