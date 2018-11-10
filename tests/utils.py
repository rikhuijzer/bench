from src.typ import System, Corpus, SystemCorpus
from src.systems.mock import timestamp_text
import shutil
import src.results

system = System('mock', Corpus.MOCK, timestamp_text, (16, ))
corpus = Corpus.MOCK
system_corpus = SystemCorpus(system, corpus)


def clear_cache():
    src.results.create_folder.cache_clear()
    src.results.create_file.cache_clear()


def cleanup():  # remove mock-MOCK folder
    """That we need this function in multiple tests only shows how annoying it is to have state in your program."""
    shutil.rmtree(str(src.results.get_folder(system_corpus)))
    clear_cache()
