from datasets.sts_benchmark import STSBenchmark
from nlu_analysers.rasa_analyser import RasaAnalyser
from nlu_converters.luis_converter import LuisConverter
from systems.dandelion import Dandelion
from pathlib import Path


# TODO: Loop over each sentence and then each system, this way the API calls do not need a sleep statement
# TODO: Integrate database to help with displaying accuracies in table including test date. Re-test monthly?


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
    # luis (also works for Rasa)
    # web_applications_corpus = Path(__file__).parent.
    luis_converter = LuisConverter()
    luis_converter.import_corpus("WebApplicationsCorpus.json")
    luis_converter.export("WebApplicationsTraining_Luis.json")

    # Rasa
    # Running via Docker possible via (sudo) docker run -p 5000:5000 rasa/rasa_nlu:latest-full
    rasa_analyser = RasaAnalyser("http://localhost:5000/parse")
    rasa_analyser.get_annotations("WebApplicationsCorpus.json", "WebApplicationsAnnotations_Rasa.json")
    rasa_analyser.analyse_annotations("WebApplicationsAnnotations_Rasa.json", "WebApplicationsCorpus.json",
                                      "WebApplicationsAnalysis_Rasa.json")


if __name__ == '__main__':
    rasa_evaluation()
