import json
import pandas
from pathlib import Path


class Corpus:
    js: dict
    df: pandas.DataFrame

    def __init__(self, corpus: str):
        file = Path(__file__).parent / 'datasets' / corpus
        with open(str(file), 'rb') as f:  # changed to rb
            self.js = json.load(f)

        pandas.set_option('max_colwidth', 180)

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


class Corpora(Corpus):
    AskUbuntuCorpus = Corpus(Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json')
    ChatbotCorpus = Corpus(Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json')
    WebApplicationsCorpus = Corpus(Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json')
