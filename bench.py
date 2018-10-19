from typing import Type

import core.utils
from systems.systems import *
import core.evaluate


# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting

# TODO: Consider also testing entities as done in evaluating paper

def analyse_system(corpus: core.utils.Corpus, system: Type[System]):
    system = system('http://0.0.0.0:5001/intents')
    corpus = core.utils.Corpus.Snips
    print(core.evaluate.classify(corpus, system))
    # print(ut._get_corpus(corpus))
    # f1_score = core.evaluate.get_f1_score(corpus, system)
    # print(f1_score)


if __name__ == '__main__':
    # TODO: Allow for easier testing of multiple benchmarks
    analyse_system(core.utils.Corpus.Snips, DeepPavlov)
