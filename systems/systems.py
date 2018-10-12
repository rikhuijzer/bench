from abc import ABC, abstractmethod
from pathlib import Path

import pandas
import rasa_nlu.training_data
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer
from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data import Message
from spacy.errors import Errors
import spacy
import subprocess


class System(ABC):

    def __init__(self, train: pandas.DataFrame):
        self.train = train

    @abstractmethod
    def train_default(self):
        pass

    def get_intent(self, sentence: str) -> str:
        pass


class Rasa(System):
    interpreter: rasa_nlu.model.Interpreter

    def __init__(self, train: pandas.DataFrame):
        super().__init__(train)

    def train_default(self):  # TODO: Update this
        # message = Message('sentence', data={'intent': 'some intent'})
        # message = Message.build('sentence', 'intent')
        training_examples = []
        for _, row in self.train.iterrows():
            training_examples.append(Message.build(row['sentence'], row['intent']))

        training_data = TrainingData(training_examples=training_examples)

        # For any intent having only one utterance it is common that sklearn gives the following warning:
        # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
        # this is no big issue according to Tom Bocklisch: https://github.com/RasaHQ/rasa_nlu/issues/288
        # training_file = rasa_nlu.training_data.loading._load(self.converter.training_file)

        # TODO: Not using config file but RasaNLUModelConfig
        # TODO: Use Spacy loading model method described below
        # import en_core_web_sm
        # nlp = en_core_web_sm.load()
        try:
            spacy.util.load_model('en_core_web_sm')
        except IOError:
            print('Spacy language model appears to be missing. Attempting to install...')
            subprocess.call(['python', '-m', 'spacy', 'download', 'en'])

        trainer = Trainer(config.load(Path(__file__).parent / 'rasa' / 'config.yml'))
        self.interpreter = trainer.train(training_data)

    def _get_classification(self, sentence):
        return self.interpreter.parse(sentence)

    def get_intent(self, sentence):
        return self._get_classification(sentence)['intent']


class DeepPavlov(System):
    bot: DefaultAgent

    def __init__(self, train: pandas.DataFrame):
        super().__init__(train)

    def train_default(self):
        # TODO: Implement non deterministic method
        self.train_deterministic()

    def train_deterministic(self):
        # some_skill = PatternMatchingSkill(responses=['intent'], patterns=['utterance 1', 'utterance 2'])
        # self.bot = DefaultAgent([skill 1, skill 2], skills_selector=HighestConfidenceSelector())
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
