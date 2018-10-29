from systems.systems import *
from core.training_data import *
from sklearn.metrics import f1_score

IntentClassifications = NamedTuple('IntentClassifications', [('system', System), ('df', pd.DataFrame)])
F1Scores = NamedTuple('F1Scores', [('system', System), ('scores', Tuple[float, ...])])


def classify_intents(system: System, corpus: Corpus) -> IntentClassifications:
    """ Run all test sentences from some corpus through system and return results. """
    df = sentences_to_dataframe(get_train_test(get_messages(corpus), TrainTest.test), Focus.intent)
    classifications = []
    for _, row in df.iterrows():
        system, classification = get_intent(system, TestSentence(row['message'], corpus))
        classifications.append(classification)

    df['classification'] = classifications
    return IntentClassifications(system, df)


def get_f1_score(system: System, corpus: Corpus, average='micro') -> F1Scores:
    """ Get f1 score for some system and corpus. Based on scikit-learn f1 score calculation. """
    system, df = classify_intents(system, corpus)
    return F1Scores(system, (f1_score(df['intent'], df['classification'], average=average)),)


def get_f1_score_runs(system: System, corpus: Corpus, n_runs: int, average='micro') -> Tuple[float, ...]:
    """ Get f1 score multiple times and re-train system each time. """
    system = System(system.name, system.knowledge, system.data + ('retrain', ))

    out = []
    for _ in range(0, n_runs):
        system, scores = get_f1_score(System(system.name, system.knowledge, system.data + ('retrain', )), corpus, average)
        out.append(scores)

    return tuple(out)
