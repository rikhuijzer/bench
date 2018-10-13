import pandas
import stringcase
from systems.systems import System
from utils import Corpus
from sklearn.metrics import f1_score


def canonical_string(intent):
    return stringcase.snakecase(intent).replace('__', '_')


def annotate_row(row):
    intent = canonical_string(row['intent'])
    classification = canonical_string(row['classification'])

    if intent == classification:
        row['true_positive'] = True
    if intent != classification and classification != 'none':
        row['false_positive'] = True
    if intent != classification and intent != 'none':
        row['false_negative'] = True
    return row


def annotate(classified_test: pandas.DataFrame) -> pandas.DataFrame:
    classified_test['true_positive'] = len(classified_test) * [False]
    classified_test['false_positive'] = len(classified_test) * [False]
    classified_test['false_negative'] = len(classified_test) * [False]

    for i, row in classified_test.iterrows():
        classified_test.loc[i] = annotate_row(row)

    return classified_test


def count_true(df, column):
    return len(df[df[column]].index)


def calculate_f1_score(annotated_test: pandas.DataFrame) -> float:
    true_positive = count_true(annotated_test, 'true_positive')
    false_positive = count_true(annotated_test, 'false_positive')
    false_negative = count_true(annotated_test, 'false_negative')
    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)
    return round(2 * ((precision * recall) / (precision + recall)), 2)


def classify(corpus: Corpus, system: System) -> pandas.DataFrame:
    test = corpus.get_test()
    classifications = []
    for _, row in test.iterrows():
        intent = system.get_intent(row['sentence'])
        classifications.append(intent)

    test['classification'] = classifications
    return test


def get_f1_score(corpus: Corpus, system: System, average='micro') -> float:
    classifications = classify(corpus, system)
    y_true = classifications['intent']
    y_pred = classifications['classification']
    return f1_score(y_true, y_pred, average=average)
