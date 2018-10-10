import warnings
from pathlib import Path

import rasa_nlu.training_data
import sklearn
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer

import nlu_converters.converter


class Rasa:
    interpreter: rasa_nlu.model.Interpreter

    def train(self, corpus):
        # For any intent having only one utterance it is common that sklearn gives the following warning:
        # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
        # this is no big issue according to Tom Bocklisch: https://github.com/RasaHQ/rasa_nlu/issues/288
        warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

        training_data = nlu_converters.converter.Converter().get_file(corpus, 'rasa.md')
        training_data = rasa_nlu.training_data.loading._load(training_data)

        trainer = Trainer(config.load(Path(__file__).parent / 'config.yml'))
        self.interpreter = trainer.train(training_data)

    def evaluate(self, sentence):
        return self.interpreter.parse(sentence)
