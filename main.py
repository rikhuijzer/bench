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
    corpora = Path(__file__).parent / 'datasets' / 'NLU-Evaluation-Corpora'
    converter = RasaConverter()
    converter.import_corpus(corpora / "WebApplicationsCorpus.json")
    converter.export('WebApplicationsCorpus')
    # rasa = Rasa()
    # rasa.train(Path(__file__).parent / 'WebApplicationsTraining_Luis.json')

    # Rasa
    # Running via Docker possible via (sudo) docker run -p 5000:5000 rasa/rasa_nlu:latest-full
    # run python -m rasa_nlu.train -c sample_configs/config_spacy.json

    # and to activate venv use: . venv/bin/activate
    # rasa_analyser = RasaAnalyser("http://localhost:5000/parse")
    # rasa_analyser.get_annotations(corpora / "WebApplicationsCorpus.json", "WebApplicationsAnnotations_Rasa.json")
    # rasa_analyser.analyse_annotations("WebApplicationsAnnotations_Rasa.json", corpora / "WebApplicationsCorpus.json",
    #                                  "WebApplicationsAnalysis_Rasa.json")


if __name__ == '__main__':
    rasa_evaluation()
