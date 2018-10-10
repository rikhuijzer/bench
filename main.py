from converters.converter_rasa import ConverterRasa
from datasets.sts_benchmark import STSBenchmark
from paths import Paths
from systems.rasa.rasa import Rasa
from corpus import Corpus


def test_sts_benchmark():
    b = STSBenchmark()
    lines = b.get_lines()
    print(lines)


def convert_rasa():
    paths = Paths('WebApplicationsCorpus', 'rasa')
    converter = ConverterRasa(paths)
    converter.import_corpus()
    converter.export()
    return converter.training_file

    # analyse(analyser_rasa, paths, rasa)


def analyse_rasa():
    paths = Paths('WebApplicationsCorpus', 'rasa')
    corpus = Corpus(paths.file_corpus())
    df = corpus.get_train()


if __name__ == '__main__':
    analyse_rasa()
