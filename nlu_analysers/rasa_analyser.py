from nlu_analysers.luis_analyser import *
from systems.rasa.rasa import Rasa


class RasaAnalyser:  # not extending luis anymore since inheritance is just plain annoying
    def __init__(self):
        self.luis_analyser = LuisAnalyser()

    def get_annotations(self, corpus, output, rasa: Rasa()):
        self.luis_analyser.get_annotations(corpus, output, method=rasa)

    def analyse_annotations(self, annotations_file, corpus_file, output_file):
        self.luis_analyser.analyse_annotations(annotations_file, corpus_file, output_file)
