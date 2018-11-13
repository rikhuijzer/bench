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
    system_corpus = get_system_corpus('rasa-spacy', tp.Corpus.SNIPS2017)
    score = get_f1_intent(system_corpus, tp.Run.NEW)
    print(score)


def dialogflow():
    system_corpus = get_system_corpus('dialogflow', tp.Corpus.WEBAPPLICATIONS)
    src.system.train(system_corpus)
    # project_id = 'bench-9bcea'
    # display_name = 'other_intent'
    # training_phrases_parts = ''
    # message_texts = ['utt1', 'utt2']
    # create_intent(project_id, display_name, training_phrases_parts, message_texts)


if __name__ == '__main__':
    import os
    # print(os.path.exists('Documents/bench.json'))
    print(os.path.exists('/bench.json'))
    logging.basicConfig(level=logging.INFO)
    dialogflow()
