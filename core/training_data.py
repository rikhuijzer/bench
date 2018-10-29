import json
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
from typing import NamedTuple

import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data.formats.markdown import MarkdownWriter
from rasa_nlu.utils import build_entity

from core.utils import *

pd.set_option('max_colwidth', 180)


class StartEnd(Enum):
    start = 0
    end = 1


class Focus(Enum):
    all = 'all'
    intent = 'intent'


class TrainTest(Enum):
    train = True
    test = False


class Corpus(Enum):
    AskUbuntu = Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json'
    Chatbot = Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json'
    WebApplications = Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json'
    Snips = Path('snips') / 'benchmark_data.json'
    Mock = ''
    Empty = ''


TestSentence = NamedTuple('Sentence', [('text', str), ('corpus', Corpus)])


def create_message(text: str, intent: str, entities: [], training: bool) -> Message:
    """ Helper function to create a message: Message used by Rasa including whether train or test sentence. """
    message = Message.build(text, intent, entities)
    message.data['training'] = training
    return message


def convert_message_to_annotated_str(message: Message) -> str:
    """ Convert Message object to string having annotated entities. """
    training_examples: List[Message] = [message]
    training_data: TrainingData = TrainingData(training_examples=training_examples)
    generated = MarkdownWriter()._generate_training_examples_md(training_data)
    generated = generated[generated.find('\n') + 3:-1]  # remove header
    return generated


def convert_nlu_evaluation_entity(text: str, entity: dict) -> dict:
    """ Convert a NLU Evaluation Corpora sentence to Entity object. See test for examples. """
    start = convert_index(text, entity['start'], StartEnd.start)
    end = convert_index(text, entity['stop'], StartEnd.end)
    return build_entity(start, end, value=entity['text'], entity_type=entity['entity'])


def sentences_to_dataframe(messages: Tuple, focus=Focus.all) -> pd.DataFrame:
    """ Returns a DataFrame (table) from a list of Message objects which can be used for visualisation.

    Args:
        messages: Sentences. Sentence is annotated if (focus != 'intent')
        focus: Focus of the DataFrame.  For intent classification visualisation choose 'intent'
    """
    data = {'message': [], 'intent': [], 'training': []}
    for message in messages:  # TODO: Apply map, filter, reduce?
        data['message'].append(message.text if focus.value == 'intent' else convert_message_to_annotated_str(message))
        data['intent'].append(message.data['intent'])
        data['training'].append(message.data['training'])
    return pd.DataFrame(data)


def convert_index(text: str, token_index: int, start_end: StartEnd) -> int:
    """ Convert token_index as used by NLU-Evaluation Corpora to character index. """
    span_generator = WordPunctTokenizer().span_tokenize(text)
    spans = [span for span in span_generator]
    return spans[token_index][start_end.value]


def read_nlu_evaluation_corpora(js: dict) -> Tuple:
    """ Convert NLU Evaluation Corpora dictionary to the internal representation. """
    out = []
    for sentence in js['sentences']:
        # TODO: Do this using map, filter, reduce?
        entities = []
        for entity in sentence['entities']:
            entities.append(convert_nlu_evaluation_entity(sentence['text'], entity))
        message = Message.build(sentence['text'], sentence['intent'], entities)
        message.data['training'] = sentence['training']
        out.append(message)
    return tuple(out)


def read_snips(js: dict) -> pd.DataFrame:
    """ Process some json file containing Snips corpus. """
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


@lru_cache(maxsize=square_ceil(len(Corpus)))
def get_messages(corpus: Corpus) -> Tuple:
    """ Get all messages: Message from some file containing corpus and cache the messages. """
    if corpus == corpus.Mock:
        return tuple(map(lambda x: create_message(str(x), 'A' if 0 <= x < 10 else 'B', [], False), range(0, 20)))

    file = Path(__file__).parent.parent / 'datasets' / corpus.value
    with open(str(file), 'rb') as f:
        js = json.load(f)

    parent_folder: Path = file.parent
    if parent_folder.name == 'NLU-Evaluation-Corpora':
        return read_nlu_evaluation_corpora(js)
    elif parent_folder.name == 'snips':
        return read_snips(js)


def get_train_test(messages: Tuple, train_test: TrainTest) -> Tuple:
    """ Get train or test split for some corpus. """
    return tuple([message for message in messages if message.data['training'] == train_test.value])
