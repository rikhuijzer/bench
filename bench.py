from typing import Type

from core import evaluate
from core.bench_utils import *
from systems.systems import *
from pathlib import Path

ROOT = Path(__file__).parent


def analyse_system(corpus: Corpus, system: Type[System]):
    system = system('http://0.0.0.0:5001/post')
    system.get_intent('test sentence')
    # print(evaluate.get_f1_score(corpus, system))


if __name__ == '__main__':
    # TODO: Allow for easier testing of multiple benchmarks
    analyse_system(Corpora.WebApplicationsCorpus, DeepPavlov)
