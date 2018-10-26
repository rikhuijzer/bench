import pandas
from systems.systems import System
from core.training_data import *


def classify_intent(corpus: Corpus, system: System) -> pandas.DataFrame:
    test = sentences_to_dataframe(get_train_test(get_messages(corpus), TrainTest.test), Focus.intent)
    classifications = []
    for _, row in test.iterrows():
        classification = system.get_intent(row['message'])
        classifications.append(classification)

    test['classification'] = classifications
    print(test)
    return test


'''
from sklearn.metrics import f1_score
def get_f1_score(corpus: Corpus, system: System, average='micro') -> float:
    classifications = classify(corpus, system)
    y_true = classifications['intent']
    y_pred = classifications['classification']
    return f1_score(y_true, y_pred, average=average)
'''
