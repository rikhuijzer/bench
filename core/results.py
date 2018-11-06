import functools
import os
import pathlib
import typing

import core.typ
import core.utils


def get_folder(sc: core.typ.SystemCorpus) -> pathlib.Path:
    return core.utils.get_root() / 'results' / '{}-{}'.format(sc.system.name, sc.corpus.name)


def get_filename(sc: core.typ.SystemCorpus, csv: core.typ.CSVs) -> pathlib.Path:
    """ Returns the filename for some system and corpus and CSV type. """
    mapping = {
        core.typ.CSVs.GENERAL: 'general.yml',
        core.typ.CSVs.INTENTS: 'intents.csv',
        core.typ.CSVs.ENTITIES: 'entities.csv'
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


def get_csv_type(csv: core.typ.CSVs) -> type:
    """ Get the corresponding type of NamedTuple for some CSV type. """
    mapping = {
        core.typ.CSVs.GENERAL: float,  # not implemented yet
        core.typ.CSVs.INTENTS: core.typ.CSVIntent,
        core.typ.CSVs.ENTITIES: float
    }
    return mapping[csv]


def convert_str_tuple(text: str, csv: core.typ.CSVs) -> typing.Any:
    """ Convert a string from csv back to NamedTuple. """
    tuple_types = get_tuple_types(get_csv_type(csv))
    converted = map(lambda t: t[0](t[1]), zip(tuple_types, text.split(',')))

    if csv == core.typ.CSVs.INTENTS:
        return core.typ.CSVIntent(*converted)
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


def write_tuple(sc: core.typ.SystemCorpus, t: typing.NamedTuple):
    """ Write some tuple to CSV. Also creates folder and file if file does not yet exist. """
    create_folder(get_folder(sc))

    if isinstance(t, core.typ.CSVIntent):
        filename = get_filename(sc, core.typ.CSVs.INTENTS)
        create_file(filename, create_header(t))
    else:
        # TODO: Accept other NamedTuples
        raise AssertionError('core.write.write_tuple got invalid input t: {}'.format(t))

    append_text(convert_tuple_str(t), filename)


def get_newest_tuple(sc: core.typ.SystemCorpus, csv: core.typ.CSVs) -> typing.Any:
    """ Get the id for the newest tuple (last line) in the csv. """
    with open(str(get_filename(sc, csv)), 'r') as f:
        content = f.read().strip()
    last_line = content[content.rfind('\n'):]
    return convert_str_tuple(last_line, csv)


def get_csv_intent(c: core.typ.Classification) -> core.typ.CSVIntent:
    """ Convert classification to intent tuple which can be sent to CSV. """
    system, corpus = c.system_corpus
    response = c.response
    # id | run | sentence | intent | classification | confidence [%] | time [ms] |


def write_classification(c: core.typ.Classification):
    system, corpus = c.system_corpus
    response = c.response

    print(2)
