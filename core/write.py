import core.typ
import pathlib
import core.utils
import typing


def get_folder(system: core.typ.System, corpus: core.typ.Corpus) -> pathlib.Path:
    return core.utils.get_root() / 'results' / '{}-{}'.format(system.name, corpus.name)


def get_filename(system: core.typ.System, corpus: core.typ.Corpus, result: core.typ.Result) -> pathlib.Path:
    return get_folder(system, corpus) / result.value


def convert_tuple(t: typing.Tuple) -> str:
    return ','.join(t)


def append_text(text: str, filename: pathlib.Path):
    with open(str(filename), 'a') as f:
        f.write(text)


def write_benchmark(system: core.typ.System, corpus: core.typ.Corpus):

