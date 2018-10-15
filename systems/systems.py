import warnings
from abc import ABC, abstractmethod

import pandas
import rasa_nlu.training_data
import sklearn
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from deeppavlov.deep import find_config
from deeppavlov.core.commands.train import train_evaluate_model_from_config


class System(ABC):

    def __init__(self, train: pandas.DataFrame):
        self.train = train

    @abstractmethod
    def train_default(self):
        pass

    def get_intent(self, sentence: str) -> str:
        pass


class Rasa(System):
    # For any intent having only one utterance it is common that sklearn gives the following warning:
    # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
    # this is no big issue according to Tom Bocklisch: https://github.com/RasaHQ/rasa_nlu/issues/288
    # training_file = rasa_nlu.training_data.loading._load(self.converter.training_file)
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

    interpreter: rasa_nlu.model.Interpreter

    def __init__(self, train: pandas.DataFrame):
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


class DeepPavlov(System):
    bot: DefaultAgent

    def __init__(self, train: pandas.DataFrame):
        super().__init__(train)

    def train_default(self):
        # TODO: Implement non deterministic method
        config = find_config('configs/classifiers/intents_snips.json')
        interact_model_by_telegram('test')
        train_evaluate_model_from_config(config)

        print(2)

    def train_deterministic(self):
        intents = self.train['intent'].unique()
        intents_utterances = {}
        for intent in intents:
            intents_utterances[intent] = []
            for _, row in self.train.iterrows():
                if row['intent'] == intent:
                    intents_utterances[intent].append(row['sentence'])

        skills = []
        for intent in intents:
            skills.append(PatternMatchingSkill(responses=[intent], patterns=intents_utterances[intent]))
        skills.append(PatternMatchingSkill(['None']))
        self.bot = DefaultAgent(skills, skills_selector=HighestConfidenceSelector())

    def get_intent(self, sentence) -> str:
        if 'bot' not in vars(self).keys():
            raise AssertionError('train system before using get_intent()')

        result = self.bot([sentence])
        if len(result) > 1:
            raise AssertionError('expected one intent result, probably not using skills_'
                                 'selector=HighestConfidenceSelector for DefaultAgent')
        return result[0]


class Watson(System):
    # possibly interesting: https://github.com/joe4k/wdcutils/

    def __init__(self, train: pandas.DataFrame):
        super().__init__(train)

    def train_default(self):
        print('unimplemented')
