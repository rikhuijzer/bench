import json
import pandas
from pathlib import Path


class Paths:
    root = Path(__file__).parent

    def __init__(self, corpus: str):
        self.corpus = corpus

    def folder_generated(self, system: str) -> Path:
        return self.root / 'generated' / system

    def _folder_corpora(self) -> Path:
        return self.root / 'datasets' / 'NLU-Evaluation-Corpora'

    def file_corpus(self) -> Path:
        return self._folder_corpora() / str(self.corpus + '.json')


class Corpus:
    js: dict
    df: pandas.DataFrame

    def __init__(self, corpus_name: str):
        file = Paths(corpus_name).file_corpus()
        with open(str(file), 'r') as f:
            self.js = json.load(f)

        pandas.set_option('max_colwidth', 140)

        data = {'sentence': [], 'intent': [], 'training': []}
        for sentence in self.js['sentences']:
            data['sentence'].append(sentence['text'])
            data['intent'].append(sentence['intent'])
            data['training'].append(sentence['training'])
        self.df = pandas.DataFrame(data)

    def get_train(self) -> pandas.DataFrame:
        return self.df.loc[self.df['training']].drop(['training'], axis=1)

    def get_test(self) -> pandas.DataFrame:
        return self.df.loc[self.df['training'] == False].drop(['training'], axis=1)
