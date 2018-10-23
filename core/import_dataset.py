import json
import re
import string
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data.formats.markdown import MarkdownWriter

pd.set_option('max_colwidth', 180)


class Corpus(Enum):
    AskUbuntu = Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json'
    Chatbot = Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json'
    WebApplications = Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json'
    Snips = Path('snips') / 'benchmark_data.json'


class Entity:
    """ Holds information about some entity in a sentence.

    For example consider the annotated sentence: Could I pay in [yen](currency)?
    In this sentence the entity 'currency' has the value 'yen'.

    To avoid duplication the entity value is not stored in this class. Hence it can only be extracted when the sentence
    is known.

    Args:
        entity: Entity name.
        start: Location of the start of the entity
        stop: Location of the end of the entity
    """

    def __init__(self, entity: str, start: int, stop: int):
        if '(' in entity or ')' in entity:
            raise ValueError('Entity contains parenthesis: ' + entity + '.')

        self.entity = entity
        self.start = start
        self.stop = stop

    def as_rasa_dict(self) -> dict:
        """ Returns dict which matches Rasa NLU entity: dict. """
        return {'start': self.start, 'end': self.stop, 'entity': self.entity, 'value': self.entity}

    def __str__(self):
        """ Returns the class as a string. Useful for debugging. """
        return 'entity: {}, start: {}, stop: {}'.format(self.entity, self.start, self.stop)


class Sentence:
    """ Holds information about sentence including intent, entities and whether train or test sentence.

    Args:
        text: Sentence text.
        intent: Intent of the sentence.
        entities: Entities occurring in sentence including their entity name
        train: Whether the sentence should be used when training
    """

    def __init__(self, text: str, intent: str, entities: List[Entity], train=True):
        self.text = text
        self.intent = intent
        self.entities = entities
        self.train = train

    def __str__(self):
        entities: List[dict] = []
        for entity in self.entities:
            entities.append(entity.as_rasa_dict())

        message = Message.build(self.text, self.intent, entities)
        training_examples: List[Message] = [message]
        training_data: TrainingData = TrainingData(training_examples=training_examples)

        generated = MarkdownWriter()._generate_training_examples_md(training_data)
        generated = generated[generated.find('\n') + 3:-1]
        generated = re.sub(r'\]\((\w|\s)*:', '](', generated)
        return generated


def find_nth(text: str, pattern: re, n: int):
    text = text.rstrip(string.punctuation)
    regex = r'(?:.*?(' + pattern + r')+){' + re.escape(str(n)) + r'}.*?((' + pattern + ')+)'
    m = re.match(regex, text)
    loc = m.span()[1] - 1
    if text[loc] != ' ':
        loc += 1
    return loc


def _nlu_evaluation_entity_converter(text: str, entity: dict) -> Entity:
    """ Convert a NLU Evaluation Corpora sentence to Entity """
    start_word_index = entity['start']
    start = find_nth(text, ' ', start_word_index)
    m = re.search(r'\W|\Z', text[start + 1:])
    end = m.start() + start + 1
    end_word_index = entity['stop']
    return Entity(entity['entity'], start, end)


'''
text = 'when is the next train in muncher freiheit?'

entity = {'entity': 'Foo', 'start': 6, 'stop': 7, 'text': 'muncher feiheit'}
expected = Entity('Foo', 25, 41)
result = import_dataset._nlu_evaluation_entity_converter(text, entity)
self.assertEqual(str(expected), str(result))
'''


def _sentences_converter(sentences: List[Sentence]) -> pd.DataFrame:
    data = {'sentence': [], 'intent': [], 'training': []}
    for sentence in sentences:
        data['sentence'].append(sentence.text)
        data['intent'].append(sentence.intent)
        data['training'].append(sentence.train)
    return pd.DataFrame(data)


def _read_nlu_evaluation_corpora(js: dict) -> List[Sentence]:
    out = []
    for sentence in js['sentences']:
        entities = []
        for entity in sentence['entities']:
            start = entity['start']
        out.append(Sentence(sentence['text'], sentence['intent'], entities, sentence['training']))

    return out


def _read_snips(js: dict) -> pd.DataFrame:
    data = {'sentence': [], 'intent': [], 'training': []}

    queries_count = 0

    for domain in js['domains']:
        for intent in domain['intents']:
            for query in intent['queries']:
                queries_count += 1
                data['sentence'].append(query['text'])
                data['intent'].append(query['results_per_service']['Snips']['classified_intent'])
                data['training'].append(False)  # TODO: Fix this

    return pd.DataFrame(data)


def _read_file(file: Path) -> pd.DataFrame:
    with open(str(file), 'rb') as f:
        js = json.load(f)

    parent_folder: Path = file.parent

    if parent_folder.name == 'NLU-Evaluation-Corpora':
        return _sentences_converter(_read_nlu_evaluation_corpora(js))
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
