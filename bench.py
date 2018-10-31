from core.evaluate import *
from core.training_data import *
import numpy as np
import logging

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting
# TODO: Consider adding training time information.
# TODO: Store output to enable re-calculating scores.


def get_f1():
    logging.info('test1')
    logging.error('test2')
    corpus = Corpus.WebApplications
    system_name = 'rasa-spacy'
    scores = get_f1_score_runs(System(system_name, Corpus.Empty, ()), corpus, n_runs=10)
    print(scores)
    print('average: {}'.format(np.mean(scores)))
    print('std: {}'.format(np.std(scores)))
    print('median: {}'.format(np.median(scores)))


def watson():
    path = Path(__file__).parent / 'generated' / 'watson' / 'ask_ubuntu.csv'
    generate_watson_intents(Corpus.AskUbuntu, path)


def lex():
    train(System('amazon-lex', Corpus.Empty, ()), Corpus.WebApplications)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    lex()
