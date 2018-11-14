import logging

import src.typ as tp
from src.evaluate import get_f1_intent


def get_system_corpus(name: str, corpus=tp.Corpus.EMPTY) -> tp.SystemCorpus:
    system = tp.System(name, corpus, timestamp='', data=())
    return tp.SystemCorpus(system, corpus)


def run():
    system_corpus = get_system_corpus('rasa-spacy', tp.Corpus.WEBAPPLICATIONS)
    score = get_f1_intent(system_corpus, tp.Run.NEW)
    print(score)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
