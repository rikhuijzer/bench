from pathlib import Path

from datasets.sts_benchmark import STSBenchmark
from nlu_analysers import rasa_analyser
from nlu_analysers.analyser import analyse
from converters.converter_rasa import ConverterRasa
from systems.rasa.rasa import Rasa
from paths import Paths


def test_sts_benchmark():
    b = STSBenchmark()
    lines = b.get_lines()
    print(lines)


def rasa_evaluation():
    paths = Paths('WebApplicationsCorpus', 'rasa')
    converter = ConverterRasa(paths)
    converter.import_corpus()
    converter.export()
    rasa = Rasa(converter)
    rasa.train()

    analyse(rasa_analyser, paths, rasa)


if __name__ == '__main__':
    rasa_evaluation()
