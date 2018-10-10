import warnings
from pathlib import Path

import rasa_nlu.training_data
import sklearn
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer


class Rasa:
    interpreter: rasa_nlu.model.Interpreter

    def __init__(self, converter):
        self.converter = converter

    def train(self):
        # For any intent having only one utterance it is common that sklearn gives the following warning:
        # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
        # this is no big issue according to Tom Bocklisch: https://github.com/RasaHQ/rasa_nlu/issues/288
        warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

        training_file = rasa_nlu.training_data.loading._load(self.converter.training_file)

        trainer = Trainer(config.load(Path(__file__).parent / 'config.yml'))
        self.interpreter = trainer.train(training_file)

    def evaluate(self, sentence):
        return self.interpreter.parse(sentence)
