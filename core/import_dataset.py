import json
import pandas as pd
from pathlib import Path
from enum import Enum
from typing import List


pd.set_option('max_colwidth', 180)


class Corpus(Enum):
    AskUbuntu = Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json'
    Chatbot = Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json'
    WebApplications = Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json'
    Snips = Path('snips') / 'benchmark_data.json'


class Entity:
    def __init__(self, entity: str, start: int, stop: int):
        self.entity = entity
        self.start = start
        self.stop = stop


class Sentence:
    def __init__(self, text: str, intent: str, entities: List[Entity]):
        self.text = text
        self.intent = intent
        self.entities = entities

    @staticmethod
    def find_nth(text: str, substring: str, n: int) -> int:
        index = 0
        for _ in range(0, n):
            index += text.find(substring, index)
        return index

    @staticmethod
    def _annotate(text: str, entity: Entity) -> str:
        words = text.split(' ')

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



