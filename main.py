from utils import *
from systems.systems import *
from typing import Type
import evaluate

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
