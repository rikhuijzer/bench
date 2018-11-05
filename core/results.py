import functools
import os
import pathlib
import typing

import core.typ
import core.utils


def get_folder(sc: core.typ.SystemCorpus) -> pathlib.Path:
    return core.utils.get_root() / 'results' / '{}-{}'.format(sc.system.name, sc.corpus.name)


def get_filename(sc: core.typ.SystemCorpus, csv: core.typ.CSVs) -> pathlib.Path:
    return get_folder(sc) / csv.value.filename


def append_text(text: str, filename: pathlib.Path):
    with open(str(filename), 'a') as f:
        f.write(text + '\n')


def convert_tuple_str(t: typing.Tuple) -> str:
    return ','.join([str(value) for value in t])


def get_names(nt: typing.NamedTuple) -> dict:
    return nt.__annotations__


def convert_str_tuple(s: str, csv: core.typ.CSVs) -> typing.NamedTuple:
    print(type(csv))
    if type(csv) == core.typ.CSVs.Intents.value.named_tuple:
        return core.typ.CSVIntent(-1, -1, 'sentence', 'intent', 'classification', -1.0, -1)


@functools.lru_cache()
def create_folder(path: pathlib) -> bool:
    path.mkdir(parents=False, exist_ok=True)
    return True


@functools.lru_cache()
def create_file(path: pathlib, header: str) -> bool:
    """ Creates file and adds header if file does not already exists. """
    if not os.path.isfile(path):
        with open(str(path), 'a+') as f:
            f.write(header + '\n')
    return True


def create_header(t: typing.NamedTuple) -> str:
    return ','.join([name for name in t.__annotations__])


def write_tuple(sc: core.typ.SystemCorpus, t: typing.NamedTuple):
    create_folder(get_folder(sc))

    if isinstance(t, core.typ.CSVIntent):
        filename = get_filename(sc, core.typ.CSVs.Intents)
        create_file(filename, create_header(t))
    else:
        # TODO: Accept other NamedTuples
        raise AssertionError('core.write.write_tuple got invalid input t: {}'.format(t))

    append_text(convert_tuple_str(t), filename)


def get_id(sc: core.typ.SystemCorpus, csv: core.typ.CSVs) -> int:
        with open(str(get_filename(sc, csv)), 'r') as f:
            content = f.read().strip()
        last_line = content[content.rfind('\n'):]



def get_csv_intent(c: core.typ.Classification) -> core.typ.CSVIntent:
    system, corpus = c.system_corpus
    response = c.response
    # id | run | sentence | intent | classification | confidence [%] | time [ms] |




def write_classification(c: core.typ.Classification):
    system, corpus = c.system_corpus
    response = c.response

    print(2)
