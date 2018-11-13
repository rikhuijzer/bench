import logging

import src.typ as tp
from src.evaluate import get_f1_intent
import src.system
from src.systems.dialogflow import create_intent

# This allows for reproducing the statistics presented in blog which is somewhat interesting


def get_system_corpus(name: str, corpus=tp.Corpus.EMPTY) -> tp.SystemCorpus:
    system = tp.System(name, corpus, timestamp='', data=())
    return tp.SystemCorpus(system, corpus)


def run():
    system_corpus = get_system_corpus('rasa-spacy', tp.Corpus.CHATBOT)
    score = get_f1_intent(system_corpus, tp.Run.NEW)
    print(score)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
