import pandas
from systems.systems import System
from utils import Corpus
from sklearn.metrics import f1_score


def classify(corpus: Corpus, system: System) -> pandas.DataFrame:
    test = corpus.get_test()
    classifications = []
    for _, row in test.iterrows():
        classification = system.get_intent(row['sentence'])
        classifications.append(classification)

    test['classification'] = classifications
    return test


def get_f1_score(corpus: Corpus, system: System, average='micro') -> float:
    classifications = classify(corpus, system)
    y_true = classifications['intent']
    y_pred = classifications['classification']
    return f1_score(y_true, y_pred, average=average)
