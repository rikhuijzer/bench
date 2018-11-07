import core.evaluate
import core.training_data
import core.typ
import numpy as np
import logging

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting
# TODO: Store output to enable re-calculating scores.


def get_f1():
    logging.info('test1')
    logging.error('test2')
    corpus = core.typ.Corpus.WebApplications
    system_name = 'rasa-spacy'
    scores = get_f1_score_runs(core.typ.System(system_name, core.typ.Corpus.WebApplications, ()), corpus, n_runs=1)
    print(scores)
    print('average: {}'.format(np.mean(scores)))
    print('std: {}'.format(np.std(scores)))
    print('median: {}'.format(np.median(scores)))


def watson():
    path = core.utils.get_root() / 'generated' / 'watson' / 'ask_ubuntu.csv'
    generate_watson_intents(core.typ.Corpus.AskUbuntu, path)


def lex():
    train(core.typ.System('amazon-lex', core.typ.Corpus.Empty, ()), core.typ.Corpus.WebApplications)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    get_f1()
