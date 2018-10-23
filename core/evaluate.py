import pandas
from systems.systems import System
from core import import_dataset
from sklearn.metrics import f1_score
from core.import_dataset import *


def classify_intent(corpus: Corpus, system: System) -> pandas.DataFrame:
    test = sentences_converter(get_test(get_corpus(corpus)), 'intent')
    classifications = []
    for _, row in test.iterrows():
        classification = system.get_intent(row['sentence'])
        classifications.append(classification)

    test['classification'] = classifications
    print(test)
    return test


def get_f1_score(corpus: import_dataset.Corpus, system: System, average='micro') -> float:
    classifications = classify(corpus, system)
    y_true = classifications['intent']
    y_pred = classifications['classification']
    return f1_score(y_true, y_pred, average=average)
