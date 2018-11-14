import logging

import src.typ as tp
from src.evaluate import evaluate, update_summary


def get_system_corpus(name: str, corpus=tp.Corpus.EMPTY) -> tp.SystemCorpus:
    system = tp.System(name, corpus, timestamp='', data=())
    return tp.SystemCorpus(system, corpus)


def run():
    system_corpus = get_system_corpus('rasa-spacy', tp.Corpus.WEBAPPLICATIONS)
    evaluate(system_corpus)
    update_summary()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    update_summary()
