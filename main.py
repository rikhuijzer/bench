from pathlib import Path

from datasets.sts_benchmark import STSBenchmark
from nlu_analysers.analyser import analyse
from nlu_converters.rasa_converter import RasaConverter
from systems.dandelion import Dandelion
from systems.rasa.rasa import Rasa
from nlu_analysers import rasa_analyser

def test_dandelion():
    s = Dandelion()

    text1 = 'Cameron wins the scar'
    text2 = 'All nominees for the Academy Awards'

    s.get_sts(text1, text2)


def test_sts_benchmark():
    b = STSBenchmark()
    lines = b.get_lines()
    print(lines)


def rasa_evaluation():
    corpus = 'WebApplicationsCorpus'
    corpora = Path(__file__).parent / 'datasets' / 'NLU-Evaluation-Corpora'
    converter = RasaConverter()
    converter.import_corpus(corpora / str(corpus + '.json'))
    converter.export(corpus)
    rasa = Rasa()
    rasa.train(corpus)

    analyser = rasa_analyser
    analyse(analyser, corpora, 'WebApplicationsAnnotations_Rasa.json', rasa)


if __name__ == '__main__':
    rasa_evaluation()
