from utils import Corpus
from systems.systems import *
from typing import Type

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
    print(system.get_intent('How can I delete my Twitter account?'))


if __name__ == '__main__':
    analyse_system(Corpus('WebApplicationsCorpus'), DeepPavlov)
