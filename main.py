from pathlib import Path

from datasets.sts_benchmark import STSBenchmark
from nlu_analysers import rasa_analyser
from nlu_analysers.analyser import analyse
from nlu_converters.rasa_converter import RasaConverter
from systems.rasa.rasa import Rasa


def test_sts_benchmark():
    b = STSBenchmark()
    lines = b.get_lines()
    print(lines)


def rasa_evaluation():
    corpus = 'WebApplicationsCorpus'
    corpora = Path(__file__).parent / 'datasets' / 'NLU-Evaluation-Corpora'
    corpus = corpora / str(corpus + '.json')
    converter = RasaConverter()
    converter.import_corpus(corpus)
    converter.export()
    rasa = Rasa(converter)
    rasa.train()

    analyse(rasa_analyser, corpus, rasa)


if __name__ == '__main__':
    rasa_evaluation()
