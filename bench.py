from core.evaluate import *
from core.training_data import *

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting

if __name__ == '__main__':
    print(get_f1_score_runs(Corpus.WebApplications, 'rasa-spacy', n_runs=10))
