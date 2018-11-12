from src.typ import System, Corpus, SystemCorpus
from src.systems.mock import get_timestamp
import shutil
import src.results


def get_system(name='mock') -> System:
    """We use different system name for each test to avoid different tests running on the same file."""
    return System(name, Corpus.MOCK, get_timestamp(), (16,))


def get_corpus() -> Corpus:
    return Corpus.MOCK


def get_system_corpus(name='mock') -> SystemCorpus:
    return SystemCorpus(get_system(name), get_corpus())


def clear_cache():
    src.results.create_folder.cache_clear()
    src.results.create_file.cache_clear()


def cleanup(name='mock'):  # remove mock-MOCK folder
    """That we need this function in multiple tests only shows how annoying it is to have state in your program."""
    shutil.rmtree(str(src.results.get_folder(get_system_corpus(name))))
    clear_cache()
