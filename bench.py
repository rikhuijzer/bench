from typing import Type

from systems.systems import *
from core.evaluate import *
from core.training_data import *

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting


def analyse_system(corpus: Corpus, system: str):
    print(get_intent(5000, system, 'hello'))
    # system = system(5000)
    # print(classify_intent(corpus, system))
    # print(ut._get_corpus(corpus))
    # f1_score = core.evaluate.get_f1_score(corpus, system)
    # print(f1_score)


if __name__ == '__main__':
    # TODO: Allow for easier testing of multiple benchmarks
    analyse_system(Corpus.WebApplications, System.Rasa)
