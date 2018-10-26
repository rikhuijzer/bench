import pandas
from systems.systems import *
from core.training_data import *
from sklearn.metrics import f1_score


def classify_intents(corpus: Corpus, system: str) -> pandas.DataFrame:
    df = sentences_to_dataframe(get_train_test(get_messages(corpus), TrainTest.test), Focus.intent)
    classifications = []
    for _, row in df.iterrows():
        classification = get_intent(system, row['message'], corpus)
        classifications.append(classification)

    df['classification'] = classifications
    return df


def get_f1_score(corpus: Corpus, system: str, average='micro') -> float:
    classifications = classify_intents(corpus, system)
    y_true = classifications['intent']
    y_pred = classifications['classification']
    return f1_score(y_true, y_pred, average=average)


def get_f1_score_runs(corpus: Corpus, system: str, average='micro', n_runs=1) -> List[float]:
    out = []
    for run in range(0, n_runs):
        out.append(get_f1_score(corpus, system, average))
        train.cache_clear()
    return out
