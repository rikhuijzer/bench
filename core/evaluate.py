from systems.systems import *
from core.training_data import *
from sklearn.metrics import f1_score

IntentClassifications = NamedTuple('IntentClassifications', [('system', System), ('df', pd.DataFrame)])


def classify_intents(system: System, corpus: Corpus) -> IntentClassifications:
    """ Run all test sentences from some corpus through system and return results. """
    df = sentences_to_dataframe(get_train_test(get_messages(corpus), TrainTest.test), Focus.intent)
    classifications = []
    for _, row in df.iterrows():
        classification = get_intent(system, TestSentence(row['message'], corpus)).classification
        classifications.append(classification)

    df['classification'] = classifications
    return IntentClassifications(system, df)


def get_f1_score(system: System, corpus: Corpus, average='micro') -> float:
    """ Get f1 score for some system and corpus. Based on scikit-learn f1 score calculation. """
    classifications = classify_intents(system, corpus).classifications
    return f1_score(classifications['intent'], classifications['classification'], average=average)


def get_f1_score_runs(system: str, corpus: Corpus, n_runs: int, average='micro') -> List[float]:
    """ Get f1 score multiple times and re-train system each time. """
    return [get_f1_score(System(system, corpus.Empty, tuple('retrain')), corpus, average) for _ in range(0, n_runs)]
