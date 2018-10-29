from core.evaluate import *
from core.training_data import *

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting


def get_f1_scores():
    corpus = Corpus.WebApplications
    system_name = 'rasa-mitie'
    print(get_f1_score_runs(System(system_name, corpus.Empty, ()), corpus, n_runs=10))


if __name__ == '__main__':
    path = Path(__file__).parent / 'generated' / 'watson' / 'web_applications.csv'
    generate_watson_intents(Corpus.WebApplications, path)
