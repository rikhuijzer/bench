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


def create_message(text: str, intent: str, entities: [], training: bool) -> Message:
    message = Message.build(text, intent, entities)
    message.data['training'] = training
    return message


def create_entity(start: int, end: int, entity: str, value: str) -> dict:
    return {'start': start, 'end': end, 'entity': entity, 'value': value}


def message_to_annotated_str(message: Message) -> str:
    training_examples: List[Message] = [message]
    training_data: TrainingData = TrainingData(training_examples=training_examples)
    generated = MarkdownWriter()._generate_training_examples_md(training_data)
    generated = generated[generated.find('\n') + 3:-1]  # remove header
    generated = re.sub(r'\]\((\w|\s)*:', '](', generated)  # fix double entity name, like train(Vehicle:Vehicle)
    return generated


def find_nth(text: str, pattern: re, n: int) -> int:
    """ Returns n-th location of some regular expression in a string. """
    if n == 0:  # this enables the regex to also find locations at start of string
        return 0
    text = text.rstrip(string.punctuation)
    regex = r'(?:.*?(' + pattern + r')+){' + str(n - 1) + r'}.*?((' + pattern + r')+)'
    m = re.match(regex, text)
    if not m:
        raise ValueError('Could not find regex with \n text: {} \n pattern: {} \n n: {}'.format(text, pattern, n))
    return m.span()[1] - 1  # minus one to convert from length to last index


def find_space(text: str, n: int) -> int:
    """ Returns character location of n-th space (or end of line) in a string. """
    loc = find_nth(text, r'\W|\Z', n)
    if text[loc] != ' ':  # the regex matched on \Z (end of line)
        loc += 1
    return loc


def tokenize(text: str, detokenize=False) -> str:
    """ Returns (de)tokenized sentence according to Microsoft LUIS. Used for working with NLU Evaluation Corpora. """
    symbols = ['.', ',', '\'', '?', '!', '&', ':', '-', '/', '(', ')']
    for symbol in symbols:
        text = text.replace(' ' + symbol + ' ', symbol) if detokenize else text.replace(symbol, ' ' + symbol + ' ')
    return text


def _nlu_evaluation_entity_converter(text: str, entity: dict) -> dict:
    """ Convert a NLU Evaluation Corpora sentence to Entity object. See test for examples. """
    def fix_tokenisation_index(tokenized: str, index: int) -> int:
        substring = tokenized[index]
        detokenized = tokenize(substring, detokenize=True)
        return len(detokenized)

    tokenized = tokenize(text)
    start_tokenized = find_space(tokenized, entity['start']) + 1
    end_tokenized = find_space(tokenized, entity['stop'] + 1)
    start = fix_tokenisation_index(tokenized, start_tokenized)
    end = fix_tokenisation_index(tokenized, end_tokenized)
    return create_entity(start, end, entity['entity'], entity['text'])


def sentences_to_dataframe(messages: List[Message], focus='all') -> pd.DataFrame:
    """ Returns a DataFrame (table) from a list of Message objects which can be used for visualisation.

    Args:
        messages: Sentences. Sentence is annotated if (focus != 'intent')
        focus: Focus of the DataFrame.  For intent classification visualisation choose 'intent'
    """
    data = {'message': [], 'intent': [], 'training': []}
    for message in messages:
        data['sentence'].append(message.text if focus == 'intent' else message_to_annotated_str(message))
        data['intent'].append(message.data['intent'])
        data['training'].append(message.data['training'])
    return pd.DataFrame(data)


def _read_nlu_evaluation_corpora(js: dict) -> List[Message]:
    """ Convert NLU Evaluation Corpora dictionary to the internal representation. """
    out = []
    for sentence in js['sentences']:
        entities = []
        for entity in sentence['entities']:
            entities.append(_nlu_evaluation_entity_converter(sentence['text'], entity))
        message = Message.build(sentence['text'], sentence['intent'], entities)
        message.data['training'] = sentence['training']
        out.append(message)
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


def _read_file(file: Path) -> List[Message]:
    with open(str(file), 'rb') as f:
        js = json.load(f)

    parent_folder: Path = file.parent

    if parent_folder.name == 'NLU-Evaluation-Corpora':
        return _read_nlu_evaluation_corpora(js)
    elif parent_folder.name == 'snips':
        return _read_snips(js)


def get_corpus(corpus: Corpus) -> List[Message]:
    return _read_file(Path(__file__).parent.parent / 'datasets' / corpus.value)


def get_train(sentences: List[Message]) -> List[Message]:
    return [sentence for sentence in sentences if sentence.data['training']]


def get_test(sentences: List[Message]) -> List[Message]:
    return [sentence for sentence in sentences if not sentence.data['training']]