import functools
import logging
import os
import pathlib
from typing import Iterable, Tuple, NamedTuple
from typing import Optional

import src.evaluate
import src.typ as tp
import src.utils


def get_folder(sc: tp.SystemCorpus) -> pathlib.Path:
    return src.utils.get_root() / 'results' / '{}-{}'.format(sc.system.name, sc.corpus.name)


def get_filename(sc: tp.SystemCorpus, csv: tp.CSVs) -> pathlib.Path:
    """ Returns the filename for some system and corpus and CSV type. """
    mapping = {
        tp.CSVs.STATS: 'general.yml',
        tp.CSVs.INTENTS: 'intents.csv',
        tp.CSVs.ENTITIES: 'entities.csv'
    }
    return get_folder(sc) / mapping[csv]


def append_text(text: str, filename: pathlib.Path) -> bool:
    """ Append text to a file, this function assumes the file exists. """
    with open(str(filename), 'a') as f:
        f.write(text + '\n')
    return True


def convert_tuple_str(t: Tuple) -> str:
    """Converting tuple to string. Removing comma's to avoid problems when reading. Good enough solution for now."""
    return ','.join([str(value).replace(',', '') for value in t])


def get_tuple_types(t: type) -> Iterable[type]:
    """ Get all the types of the fields in some NamedTuple. """
    return map(lambda x: x[1], t._field_types.items())


def get_csv_type(csv: tp.CSVs) -> type:
    """ Get the corresponding type of NamedTuple for some CSV type. """
    mapping = {
        tp.CSVs.STATS: tp.CSVStats,
        tp.CSVs.INTENTS: tp.CSVIntent,
        tp.CSVs.ENTITIES: tp.CSVEntity
    }
    return mapping[csv]


def convert_str_tuple(text: str, csv: tp.CSVs) -> tp.CSV_types:
    """ Convert a string from csv back to NamedTuple. """
    tuple_types = get_tuple_types(get_csv_type(csv))
    converted = map(lambda t: t[0](t[1]), zip(tuple_types, text.split(',')))

    if csv == tp.CSVs.INTENTS:
        return tp.CSVIntent(*converted)
    elif csv == tp.CSVs.ENTITIES:
        return tp.CSVEntity(*converted)
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
    elif csv == tp.CSVEntity:
        filename = get_filename(sc, tp.CSVs.ENTITIES)
        create_file(filename, create_header(csv))
    else:
        # TODO: Accept other NamedTuples
        raise AssertionError('src.write.write_tuple got invalid input t: {}'.format(csv))
    return filename


def write_tuple(sc: tp.SystemCorpus, namedtuple: NamedTuple) -> bool:
    """ Write some tuple to CSV. Also creates folder and file if file does not yet exist. """
    logging.info('Writing {}'.format(namedtuple))

    filename = initialize_file(sc, type(namedtuple))
    return append_text(convert_tuple_str(namedtuple), filename)


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
                        gold_standard=classification.message.data['intent'],
                        classification=classification.response.intent,
                        confidence=classification.response.confidence,
                        time=0)


def get_csv_entity(classification: tp.Classification, source: str, entity: dict, intent_id: int) -> tp.CSVEntity:
    system_corpus = tp.SystemCorpus(classification.system, classification.message.data['corpus'])
    newest_tuple = get_newest_tuple(system_corpus, tp.CSVs.ENTITIES)
    return tp.CSVEntity(id=newest_tuple.id + 1 if newest_tuple else 0,
                        intent_id=intent_id,
                        timestamp=classification.system.timestamp,
                        source=source,
                        entity=entity['entity'],
                        value=entity['value'],
                        start=entity['start'],
                        end=entity['end'],
                        confidence=entity['confidence'] if 'confidence' in entity else -1.0)


def write_classification(classification: tp.Classification):
    """Unpack and write a classification to various files."""
    system_corpus = tp.SystemCorpus(classification.system, classification.message.data['corpus'])

    csv_intent = get_csv_intent(classification)
    write_tuple(system_corpus, csv_intent)

    data = classification.message.data
    if 'entities' in data:
        def write_entity(entity: dict, source: str) -> bool:
            csv_entity = get_csv_entity(classification, source, entity, csv_intent.id)
            return write_tuple(system_corpus, csv_entity)

        def write_entities(source: str, entities: Iterable[dict]) -> Iterable[bool]:
            return tuple(map(lambda entity: write_entity(entity, source), entities))

        write_entities('gold standard', data['entities'])
        write_entities('classification', classification.response.entities)


def get_elements(sc: tp.SystemCorpus, csv: tp.CSVs) -> Iterable[tp.CSV_types]:
    """Get all elements from some results CSV file."""
    content = read_file(sc, csv)
    lines = content.strip().split('\n')

    return map(lambda line: convert_str_tuple(line, csv), lines[1:])
