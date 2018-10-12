import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import pandas
import rasa_nlu.training_data
import sklearn
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer


class System(ABC):
    @abstractmethod
    def __init__(self, train: pandas.DataFrame):
        pass

    @abstractmethod
    def get_intent(self, sentence: str) -> str:
        pass


class Rasa(System):
    interpreter: rasa_nlu.model.Interpreter

    def __init__(self, train: pandas.DataFrame):
        super().__init__(train)

        self.converter = train  # converter

        # For any intent having only one utterance it is common that sklearn gives the following warning:
        # UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
        # this is no big issue according to Tom Bocklisch: https://github.com/RasaHQ/rasa_nlu/issues/288
        warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

        training_file = rasa_nlu.training_data.loading._load(self.converter.training_file)

        trainer = Trainer(config.load(Path(__file__).parent / 'config.yml'))
        self.interpreter = trainer.train(training_file)

    def _get_classification(self, sentence):
        return self.interpreter.parse(sentence)

    def get_intent(self, sentence):
        return self._get_classification(sentence)['intent']


class DeepPavlov(System):
    bot: DefaultAgent

    def __init__(self, train: pandas.DataFrame):
        super().__init__(train)

        # some_skill = PatternMatchingSkill(responses=['intent'], patterns=['utterance 1', 'utterance 2'])
        # self.bot = DefaultAgent([skill 1, skill 2], skills_selector=HighestConfidenceSelector())
        intents = train['intent'].unique()
        intents_utterances = {}
        for intent in intents:
            intents_utterances[intent] = []
            for _, row in train.iterrows():
                if row['intent'] == intent:
                    intents_utterances[intent].append(row['sentence'])

        skills = []
        for intent in intents:
            skills.append(PatternMatchingSkill(responses=[intent], patterns=intents_utterances[intent]))
        self.bot = DefaultAgent(skills, skills_selector=HighestConfidenceSelector())

    def get_intent(self, sentence) -> str:
        result = self.bot([sentence])
        if len(result) > 1:
            raise AssertionError('Expected one intent result')
        return result[0]
