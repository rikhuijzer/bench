from pathlib import Path

from rasa_nlu import config
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer
import rasa_nlu.training_data
from rasa_nlu import utils
import json


class Rasa:
    def train(self, training_data):
        training_data = rasa_nlu.training_data.loading._load(training_data)

        for examples in training_data.training_examples:
            print(examples)

        trainer = Trainer(config.load(Path(__file__).parent / 'config.yml'))
        trainer.train(training_data)
        model_directory = trainer.persist('./projects/default/')  # Returns the directory the model is stored in
        print('model directory: ' + model_directory)

        # where model_directory points to the model folder
        interpreter = Interpreter.load(model_directory)
        interpreter.parse(u"The text I want to understand")
