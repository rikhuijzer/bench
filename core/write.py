import functools
import pathlib
import typing

import core.typ
import core.utils


def get_folder(sc: core.typ.SystemCorpus) -> pathlib.Path:
    return core.utils.get_root() / 'results' / '{}-{}'.format(sc.system.name, sc.corpus.name)


def get_filename(sc: core.typ.SystemCorpus, csvs: core.typ.CSVs) -> pathlib.Path:
    return get_folder(sc) / csvs.value


def convert_tuple(t: typing.Tuple) -> str:
    return ','.join([str(value) for value in t])


def append_text(text: str, filename: pathlib.Path):
    with open(str(filename), 'a') as f:
        f.write(text + '\n')


@functools.lru_cache()
def create_folder(path: pathlib) -> bool:
    path.mkdir(parents=False, exist_ok=True)
    return True


@functools.lru_cache()
def create_file(path: pathlib, header: str) -> bool:
    with open(str(path), 'a+') as f:
        f.write(header + '\n')
    return True


def create_header(t: typing.NamedTuple) -> str:
    return ','.join([name for name in t.__annotations__])


def write_tuple(sc: core.typ.SystemCorpus, t: typing.NamedTuple):
    create_folder(get_folder(sc))

    # check file exists
    if isinstance(t, core.typ.CSVIntent):
        filename = get_filename(sc, core.typ.CSVs.Intents)
        create_file(filename, create_header(t))
    else:
        # TODO: Accept other NamedTuples
        raise AssertionError('core.write.write_tuple got invalid input t: {}'.format(t))

    append_text(convert_tuple(t), filename)
