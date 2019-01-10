import logging

import src.typ as tp
from src.evaluate import evaluate


def get_system_corpus(name: str, corpus=tp.Corpus.EMPTY) -> tp.SystemCorpus:
    system = tp.System(name, corpus, timestamp='', data=())
    return tp.SystemCorpus(system, corpus)


def run():
    # the first parameter of get_system_corpus specifies the system to be benchmarked.
    # this string should match a name in the Docker compose file or a name listed in get_system_corpus
    #
    # the second parameter specifies the corpus to be used for the benchmark
    # see tp.Corpus for the possible options
    system_corpus = get_system_corpus('rasa-mitie', tp.Corpus.WEBAPPLICATIONS)
    evaluate(system_corpus)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
