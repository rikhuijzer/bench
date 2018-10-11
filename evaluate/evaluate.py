import pandas as pd
import stringcase


def canonical_string(intent):
    intent = stringcase.snakecase(intent)
    intent = intent.replace('__', '_')
    return intent


def _annotate_row(row):
    intent = canonical_string(row['intent'])
    classification = canonical_string(row['classification'])

    if intent == classification:
        row['true_positive'] = True
    if intent != classification and classification != 'none':
        row['false_positive'] = True
    if intent != classification and intent != 'none':
        row['false_negative'] = True
    return row


def annotate(df: pd.DataFrame):
    df['true_positive'] = len(df) * [False]
    df['false_positive'] = len(df) * [False]
    df['false_negative'] = len(df) * [False]

    for i, row in df.iterrows():
        df.loc[i] = _annotate_row(row)

    return df


def _count_true(df, column):
    count = 0
    for i, row in df.iterrows():
        if row[column]:
            count += 1
    return count


def f1_score(df):
    true_positive = _count_true(df, 'true_positive')
    false_positive = _count_true(df, 'false_positive')
    false_negative = _count_true(df, 'false_negative')
    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)
    return round(2 * ((precision * recall) / (precision + recall)), 2)
