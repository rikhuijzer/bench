# from utils.telegram_utils.telegram_ui import interact_model_by_telegram
from utils.telegram_utils.telegram_ui import interact_model_by_telegram
from typing import Type

from core import evaluate
from core.utils import *
from systems.systems import *


'''
def analyse_():
    rasa = Rasa(convert_rasa())
    paths = Paths('WebApplicationsCorpus', 'rasa')
    corpus = Corpus(paths.file_corpus())
    df = evaluate.classify(corpus.get_test(), rasa)
    print(df)
    print(evaluate.f1_score(df))
'''


def analyse_system(corpus: Corpus, system: Type[System]):
    system = system(corpus.get_train())
    system.train_default()
    print(evaluate.get_f1_score(corpus, system))


if __name__ == '__main__':
    # TODO: Allow for easier testing of multiple benchmarks
    analyse_system(Corpora.WebApplicationsCorpus, DeepPavlov)
