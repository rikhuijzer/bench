import logging

import src.typ as tp
from src.evaluate import get_f1_intent
import src.system


# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting

def get_system_corpus(name: str) -> tp.SystemCorpus:
    system = tp.System(name, tp.Corpus.EMPTY, timestamp='', data=())
    corpus = tp.Corpus.ASKUBUNTU
    return tp.SystemCorpus(system, corpus)


def run():
    system_corpus = get_system_corpus('rasa-spacy')
    score = get_f1_intent(system_corpus, tp.Run.NEW)
    print(score)


def dialogflow():
    system_corpus = get_system_corpus('dialogflow')
    src.system.train(system_corpus)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dialogflow()
