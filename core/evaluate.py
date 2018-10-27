import pandas as pd
from systems.systems import *
from core.training_data import *
from sklearn.metrics import f1_score


IntentClassifications = NamedTuple('IntentClassifications', [('system', System), ('classifications', pd.DataFrame)])
# System = NamedTuple('System', [('name', str), ('knowledge', Corpus)])


def classify_intents(system: System, corpus: Corpus) -> IntentClassifications:
    """ Run all test sentences from some corpus through system and return results. """
    df = sentences_to_dataframe(get_train_test(get_messages(corpus), TrainTest.test), Focus.intent)
    classifications = []
    for _, row in df.iterrows():
        classification = get_intent(system, TestSentence(row['message'], corpus))
        classifications.append(classification)

    df['classification'] = classifications
    return IntentClassifications(system, df)


def get_f1_score(system: str, corpus: Corpus, system_knowledge: Corpus, average='micro') -> float:
    """ Get f1 score for some system and corpus. Based on scikit-learn f1 score calculation. """
    classifications = classify_intents(system, corpus).classifications
    classifications.classifications
    y_true = classifications['intent']
    y_pred = classifications['classification']
    return f1_score(y_true, y_pred, average=average)


def get_f1_score_runs(system: str, corpus: Corpus, n_runs: int, average='micro') -> List[float]:
    """ Get f1 score multiple times and re-train system each time. """
    out = []
    for run in range(0, n_runs):
        train(system, corpus)
        out.append(get_f1_score(system, corpus, corpus, average))
    return out
