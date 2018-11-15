import src.typ as tp
from src.systems.mock import get_timestamp
import shutil
import src.results
from typing import Callable, Iterable, Any, TypeVar
from src.evaluate import write_classifications


def get_system(name='mock') -> tp.System:
    """We use different system name for each test to avoid different tests running on the same file."""
    return tp.System(name, tp.Corpus.MOCK, get_timestamp(), (16,))


def get_corpus() -> tp.Corpus:
    return tp.Corpus.MOCK


def get_system_corpus(name='mock') -> tp.SystemCorpus:
    return tp.SystemCorpus(get_system(name), get_corpus())


def run_with_file_operations(name_suffix: str, func: Callable[[tp.SystemCorpus], tp.T]) -> tp.T:
    """Creates new results subfolder, runs func and cleans results subfolder.
    This complexity is brought to you by functions with side-effects (specifically, filesystem operations)."""
    import warnings
    warnings.filterwarnings('ignore')

    name = 'mock_' + name_suffix
    system_corpus = get_system_corpus(name)
    tuple(write_classifications(system_corpus))
    result = func(system_corpus)
    cleanup(name)
    return result


def clear_cache():
    src.results.create_folder.cache_clear()
    src.results.create_file.cache_clear()


def cleanup(name='mock'):
    """That we need this function in multiple tests only shows how annoying it is to have state in your program."""
    shutil.rmtree(str(src.results.get_folder(get_system_corpus(name))))
    clear_cache()
