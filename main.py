from converters.converter_rasa import ConverterRasa
from datasets.sts_benchmark import STSBenchmark
from paths import Paths
from systems.rasa.rasa import Rasa
from corpus import Corpus
from analysers import analyser_rasa


def test_sts_benchmark():
    b = STSBenchmark()
    lines = b.get_lines()
    print(lines)


def convert_rasa():
    paths = Paths('WebApplicationsCorpus', 'rasa')
    converter = ConverterRasa(paths)
    converter.import_corpus()
    converter.export()
    return converter

    # analyse(analyser_rasa, paths, rasa)


def analyse_rasa():
    rasa = Rasa(convert_rasa())
    paths = Paths('WebApplicationsCorpus', 'rasa')
    corpus = Corpus(paths.file_corpus())
    analyser_rasa.test(corpus, rasa)


if __name__ == '__main__':
    analyse_rasa()
