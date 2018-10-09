from datasets.sts_benchmark import STSBenchmark
from nlu_analysers.rasa_analyser import RasaAnalyser
from nlu_converters.rasa_converter import RasaConverter
from nlu_converters.luis_converter import LuisConverter
from systems.dandelion import Dandelion
from pathlib import Path
from systems.rasa.rasa import Rasa


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

    rasa_analyser = RasaAnalyser()
    annotations_file = 'WebApplicationsAnnotations_Rasa.json'
    rasa_analyser.get_annotations(corpus=corpora / "WebApplicationsCorpus.json", output=annotations_file, rasa=rasa)
    rasa_analyser.analyse_annotations(annotations_file, corpora / "WebApplicationsCorpus.json",
                                      "WebApplicationsAnalysis_Rasa.json")


if __name__ == '__main__':
    rasa_evaluation()
