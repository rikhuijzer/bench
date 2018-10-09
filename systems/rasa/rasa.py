from pathlib import Path

import rasa_nlu.training_data
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer

import nlu_converters.converter


class Rasa:
    interpreter: rasa_nlu.model.Interpreter

    def train(self, corpus):
        training_data = nlu_converters.converter.Converter().get_file(corpus, 'rasa.md')
        training_data = rasa_nlu.training_data.loading._load(training_data)

        trainer = Trainer(config.load(Path(__file__).parent / 'config.yml'))
        self.interpreter = trainer.train(training_data)

    def evaluate(self, sentence):
        self.interpreter.parse(sentence)
