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
        """
        :return: Sentence with annotated entities. This does not return self.intent or self.train information.
        """
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


def find_nth(text: str, pattern: re, n: int) -> int:
    """ Returns n-th location of some regular expression in a string. See test for examples. """
    text = text.rstrip(string.punctuation)
    regex = r'(?:.*?(' + pattern + r')+){' + re.escape(str(n)) + r'}.*?((' + pattern + ')+)'
    m = re.match(regex, text)
    # print('text: {}, pattern: {}, m: {}'.format(text, pattern, m))
    if m:
        loc = m.span()[1] - 1  # the span returns len(match: str) not the last index of match: str
        if text[loc] != ' ':  # regex usually matches on string plus some space, we add one to the index if
            loc += 1
    else:  # hacking around the inconsistently formatted data
        loc = -1
    return loc


def luis_tokenizer(text: str, detokenize=False) -> str:
    """ Returns (de)tokenized sentence in Microsoft LUIS method. Used for working with NLU Evaluation Corpora. """
    symbols = ['.', ',', '\'', '?', '!', '&', ':', '-', '/', '(', ')']
    for symbol in symbols:
        text = text.replace(' ' + symbol + ' ', symbol) if detokenize else text.replace(symbol, ' ' + symbol + ' ')
    return text


def _nlu_evaluation_entity_converter(text: str, entity: dict) -> Entity:
    """ Convert a NLU Evaluation Corpora sentence to Entity object. See test for examples. """
    start_word_index = entity['start']
    start = find_nth(text, r'\W', start_word_index - 1) + 1
    if start == -1:  # hacking around the inconsistently formatted data
        start = text.find(entity['text'])
    end = start + len(entity['text'])
    return Entity(entity['entity'], start, end)


def _sentences_converter(sentences: List[Sentence]) -> pd.DataFrame:
    """ Convert a list of Sentence objects into a pd.DataFrame which can be used for visualisation. """
    data = {'sentence': [], 'intent': [], 'training': []}
    for sentence in sentences:
        data['sentence'].append(sentence.text)
        data['intent'].append(sentence.intent)
        data['training'].append(sentence.train)
    return pd.DataFrame(data)


def _read_nlu_evaluation_corpora(js: dict) -> List[Sentence]:
    """ Convert NLU Evaluation Corpora dictionary to the internal representation. """
    out = []
    for sentence in js['sentences']:
        entities = []
        for entity in sentence['entities']:
            entities.append(_nlu_evaluation_entity_converter(sentence['text'], entity))
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
