from core.evaluate import *
from core.training_data import *
import numpy as np
import logging

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting
# TODO: Consider adding training time information.


def get_f1():
    logging.info('test1')
    logging.error('test2')
    corpus = Corpus.WebApplications
    system_name = 'watson'
    scores = get_f1_score_runs(System(system_name, corpus.WebApplications, ()), corpus, n_runs=1)
    print(scores)
    print('average: {}'.format(np.mean(scores)))
    print('std: {}'.format(np.std(scores)))
    print('median: {}'.format(np.median(scores)))


def watson():
    path = Path(__file__).parent / 'generated' / 'watson' / 'web_applications.csv'
    generate_watson_intents(Corpus.WebApplications, path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    get_f1()
