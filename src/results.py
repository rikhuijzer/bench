import functools
import os
import pathlib
import typing

import src.typ
import src.utils
import src.evaluate
import logging


def get_folder(sc: src.typ.SystemCorpus) -> pathlib.Path:
    return src.utils.get_root() / 'results' / '{}-{}'.format(sc.system.name, sc.corpus.name)


def get_filename(sc: src.typ.SystemCorpus, csv: src.typ.CSVs) -> pathlib.Path:
    """ Returns the filename for some system and corpus and CSV type. """
    mapping = {
        src.typ.CSVs.GENERAL: 'general.yml',
        src.typ.CSVs.INTENTS: 'intents.csv',
        src.typ.CSVs.ENTITIES: 'entities.csv'
    }
    return get_folder(sc) / mapping[csv]


def append_text(text: str, filename: pathlib.Path):
    """ Append text to a file, this function assumes the file exists. """
    with open(str(filename), 'a') as f:
        f.write(text + '\n')


def convert_tuple_str(t: typing.Tuple) -> str:
    return ','.join([str(value) for value in t])


def get_tuple_types(t: type) -> typing.Iterable[type]:
    """ Get all the types of the fields in some NamedTuple. """
    return map(lambda x: x[1], t._field_types.items())


def get_csv_type(csv: src.typ.CSVs) -> type:
    """ Get the corresponding type of NamedTuple for some CSV type. """
    mapping = {
        src.typ.CSVs.GENERAL: src.typ.CSVGeneral,
        src.typ.CSVs.INTENTS: src.typ.CSVIntent,
        src.typ.CSVs.ENTITIES: src.typ.CSVEntity
    }
    return mapping[csv]


def convert_str_tuple(text: str, csv: src.typ.CSVs) -> src.typ.CSV_types:
    """ Convert a string from csv back to NamedTuple. """
    tuple_types = get_tuple_types(get_csv_type(csv))
    converted = map(lambda t: t[0](t[1]), zip(tuple_types, text.split(',')))

    if csv == src.typ.CSVs.INTENTS:
        return src.typ.CSVIntent(*converted)
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


def create_header(t: typing.NamedTuple) -> str:
    """ Create a header which can be placed in first line of CSV. """
    return ','.join([name for name in t.__annotations__])


def write_tuple(sc: src.typ.SystemCorpus, namedtuple: typing.NamedTuple):
    """ Write some tuple to CSV. Also creates folder and file if file does not yet exist. """
    logging.info('Writing: {}'.format(namedtuple))

    create_folder(get_folder(sc))

    if isinstance(namedtuple, src.typ.CSVIntent):
        filename = get_filename(sc, src.typ.CSVs.INTENTS)
        create_file(filename, create_header(namedtuple))
    else:
        # TODO: Accept other NamedTuples
        raise AssertionError('src.write.write_tuple got invalid input t: {}'.format(namedtuple))

    append_text(convert_tuple_str(namedtuple), filename)



def get_newest_tuple(sc: src.typ.SystemCorpus, csv: src.typ.CSVs) -> typing.Optional[src.typ.CSV_types]:
    """ Get the id for the newest tuple (last line) in the csv. """
    filename = get_filename(sc, csv)

    if not os.path.isfile(filename):
        return None

    with open(str(filename), 'r') as f:
        content = f.read().strip()
    last_line = content[content.rfind('\n'):]
    return convert_str_tuple(last_line, csv)


def get_csv_intent(classification: src.typ.Classification) -> src.typ.CSVIntent:
    """ Convert classification to intent tuple which can be sent to CSV. """
    system_corpus = src.typ.SystemCorpus(classification.system, classification.message.data['corpus'])
    newest_tuple = get_newest_tuple(system_corpus, src.typ.CSVs.INTENTS)
    return src.typ.CSVIntent(id=newest_tuple.id + 1 if newest_tuple else 0,
                             run=classification.system.timestamp,
                             sentence=classification.message.text,
                             intent=classification.message.data['intent'],
                             classification=classification.response.intent,
                             confidence=classification.response.confidence,
                             time=0)


def write_classification(classification: src.typ.Classification):
    """Unpack and write a classification to various files."""
    csv_intent = get_csv_intent(classification)
    system_corpus = src.typ.SystemCorpus(classification.system, classification.message.data['corpus'])
    write_tuple(system_corpus, csv_intent)
