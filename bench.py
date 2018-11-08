import logging
from src.evaluate import run_bench
import src.typ as tp

# TODO: Figure out what dataset is http://files.deeppavlov.ai/datasets/snips_intents/train.csv of about 16036 entries
# This allows for reproducing the statistics presented in blog which is somewhat interesting
# TODO: Store output to enable re-calculating scores.


def run_rasa():
    system = tp.System('rasa-spacy', tp.Corpus.EMPTY, timestamp='', data=())
    corpus = tp.Corpus.WEBAPPLICATIONS
    run_bench(tp.SystemCorpus(system, corpus))  # TODO: Add number of runs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_rasa()
