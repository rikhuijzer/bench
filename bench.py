from typing import Type

import core.utils as ut
from systems.systems import *
import core.evaluate


def analyse_system(corpus: ut.Corpus, system: Type[System]):
    system = system('http://0.0.0.0:5001/intents')
    corpus = ut.Corpus.Snips
    print(ut._get_corpus(corpus))
    # f1_score = core.evaluate.get_f1_score(corpus, system)
    # print(f1_score)


if __name__ == '__main__':
    # TODO: Allow for easier testing of multiple benchmarks
    analyse_system(ut.Corpus.Snips, DeepPavlov)
