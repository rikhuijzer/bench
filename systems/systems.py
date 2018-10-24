import warnings
import json
from abc import ABC, abstractmethod

import requests
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
import pandas as pd
import rasa_nlu
import sklearn


class System(ABC):
    HEADER_JSON = {'content-type': 'application/json'}
    port: int

    def __init__(self, port: int):
        self.port = port

    @abstractmethod
    def get_intent(self, sentence: str) -> str:
        pass


'''
class Rasa(System):
    # For any intent having only one utterance it is common that sklearn gives the following warning:
    # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
    # this is no big issue according to Tom Bocklisch: https://github.com/RasaHQ/rasa_nlu/issues/288
    # training_file = rasa_nlu.training_data.loading._load(self.converter.training_file)
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

    interpreter: rasa_nlu.model.Interpreter

    def __init__(self, train: pd.DataFrame):
        super().__init__(train)

    def train_default(self):
        configuration_values = {
            'language': 'en_core_web_sm',
            'pipeline': [{'name': 'nlp_spacy'},
                         {'name': 'tokenizer_spacy'},
                         {'name': 'ner_crf'},
                         {'name': 'intent_featurizer_spacy'},
                         {'name': 'intent_classifier_sklearn'},
                         {'name': 'ner_synonyms'}],
            'data': None
        }
        trainer = Trainer(RasaNLUModelConfig(configuration_values))

        training_examples = []
        for _, row in self.train.iterrows():
            training_examples.append(Message.build(row['sentence'], row['intent']))
        training_data = TrainingData(training_examples=training_examples)

        self.interpreter = trainer.train(training_data)

    def _get_classification(self, sentence):
        return self.interpreter.parse(sentence)

    def get_intent(self, sentence):
        return self._get_classification(sentence)['intent']['name']
'''


class Rasa(System):
    def __init__(self, port: int):
        super().__init__(port)

    def get_intent(self, sentence: str):
        data = {'q': sentence}
        r = requests.post('localhost:{}/parse'.format(self.port), data=json.dumps(data), headers=self.HEADER_JSON)
        return r.json()[0][0][0]


class DeepPavlov(System):
    def __init__(self, port: int):
        super().__init__(port)

    def get_intent(self, sentence: str):
        data = {'context': [sentence]}
        r = requests.post('localhost:{}/intent'.format(self.port), data=json.dumps(data), headers=self.HEADER_JSON)
        return r.json()[0][0][0]


class Watson(System):
    # possibly interesting: https://github.com/joe4k/wdcutils/
    def __init__(self, url):
        super().__init__(url)

    def get_intent(self, sentence: str):
        print('unimplemented')
