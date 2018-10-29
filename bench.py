from core.evaluate import *
from core.training_data import *

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting

if __name__ == '__main__':
    corpus = Corpus.WebApplications
    system_name = 'rasa-mitie'
    print(get_f1_score_runs(System(system_name, corpus.Empty, ()), corpus, n_runs=10))
