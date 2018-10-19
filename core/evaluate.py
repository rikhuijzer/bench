import pandas
from systems.systems import System
import core.utils
from sklearn.metrics import f1_score


def classify(corpus: core.utils.Corpus, system: System) -> pandas.DataFrame:
    test = core.utils.get_test(corpus)
    classifications = []
    for _, row in test.iterrows():
        classification = system.get_intent(row['sentence'])
        classifications.append(classification)

    test['classification'] = classifications
    print(test)
    return test


def get_f1_score(corpus: core.utils.Corpus, system: System, average='micro') -> float:
    classifications = classify(corpus, system)
    y_true = classifications['intent']
    y_pred = classifications['classification']
    return f1_score(y_true, y_pred, average=average)
