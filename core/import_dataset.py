import json
import pandas as pd
from pathlib import Path
from enum import Enum
from typing import List
import re

pd.set_option('max_colwidth', 180)


class Corpus(Enum):
    AskUbuntu = Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json'
    Chatbot = Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json'
    WebApplications = Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json'
    Snips = Path('snips') / 'benchmark_data.json'


class Entity:
    """ Holds information about some entity in a sentence.

    For example consider the sentence: Could I pay in [yen](currency)?
    In this sentence the entity 'currency' has the value 'yen'.

    To avoid duplication the entity value is not stored in this class. Hence it can only be extracted when the sentence
    is known.

    Args:
        entity: Entity name.
        start: Which word in the sentence is the start of the entity. Count starts at zero.
        stop: Which word in the sentence is the end of the entity. Count starts at zero.
    """

    def __init__(self, entity: str, start: int, stop: int):
        if '(' in entity or ')' in entity:
            raise ValueError('Entity contains parenthesis: ' + entity + '.')

        self.entity = entity
        self.start = start
        self.stop = stop


class Sentence:
    def __init__(self, text: str, intent: str, entities: List[Entity]):
        self.text = text
        self.intent = intent
        self.entities = entities

    @staticmethod
    def _increase_index_annotated_sentence(text: str, n: int) -> int:
        """ Fix the index as specified by Entity class when annotations have been added. """
        # I have 50 [yen](currency lorem ipsum) in my pocket
        tokens = text.split(' ')
        for i, token in enumerate(tokens):
            if '](' in token:
                for j in range(i, len(tokens)):
                    if ')' in tokens[j]:
                        n += j - i + 1
                        break
        return n

    @staticmethod
    def find_nth(text: str, substring: str, n: int) -> int:
        if n == 1:
            return text.find(substring)
        else:
            return text.find(substring, Sentence.find_nth(text, substring, n - 1) + 1)

    @staticmethod
    def _annotate(text: str, entity: Entity) -> str:
        left_bracket = Sentence.find_nth(text, ' ', entity.start) + 1  # TODO: Use increase_index
        text = text[:left_bracket] + '(' + text[left_bracket:]
        # [(m.start(0), m.end(0)) for m in re.finditer(r'', text)]
        # right_bracket = text.find(re.findall(r'', ))
        right_bracket = 3
        # TODO: Find all entity values in sentence. Create replacer and replace entity value with annotated version.
        return text

    def __str__(self):
        out = self.text
        for entity in self.entities:
            out = self._annotate(out, entity)
        return '{' + self.intent + '} ' + self.text


def _read_nlu_evaluation_corpora(js: dict) -> pd.DataFrame:
    data = {'sentence': [], 'intent': [], 'training': []}
    for sentence in js['sentences']:
        data['sentence'].append(sentence['text'])
        data['intent'].append(sentence['intent'])
        data['training'].append(sentence['training'])
    return pd.DataFrame(data)


def _read_snips(js: dict) -> pd.DataFrame:
    data = {'sentence': [], 'intent': [], 'training': []}

    queries_count = 0

    for domain in js['domains']:
        for intent in domain['intents']:
            for query in intent['queries']:
                queries_count += 1
                data['sentence'].append(query['text'])
                data['intent'].append(query['results_per_service']['Snips']['classified_intent'])
                data['training'].append(False)  # TODO: Improve this

    return pd.DataFrame(data)


def _read_file(file: Path) -> pd.DataFrame:
    with open(str(file), 'rb') as f:
        js = json.load(f)

    parent_folder: Path = file.parent

    if parent_folder.name == 'NLU-Evaluation-Corpora':
        return _read_nlu_evaluation_corpora(js)
    elif parent_folder.name == 'snips':
        return _read_snips(js)


def _get_corpus(corpus: Corpus) -> pd.DataFrame:
    return _read_file(Path(__file__).parent.parent / 'datasets' / corpus.value)


def _get_train(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['training']].drop(['training'], axis=1).reset_index(drop=True)


def _get_test(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['training'] == False].drop(['training'], axis=1).reset_index(drop=True)


def get_train(corpus: Corpus) -> pd.DataFrame:
    return _get_train(_get_corpus(corpus))


def get_test(corpus: Corpus) -> pd.DataFrame:
    return _get_test(_get_corpus(corpus))



