import json
import pandas


class Corpus:
    js: json.load
    df: pandas.DataFrame

    def __init__(self, file):
        with open(file, 'r') as f:
            self.js = json.load(f)

        pandas.set_option('max_colwidth', 140)

        data = {'sentence': [], 'intent': [], 'training': []}
        for sentence in self.js['sentences']:
            data['sentence'].append(sentence['text'])
            data['intent'].append(sentence['intent'])
            data['training'].append(sentence['training'])
        self.df = pandas.DataFrame(data)

    def get_train(self):
        return self.df.loc[self.df['training']].drop(['training'], axis=1)

    def get_test(self):
        return self.df.loc[self.df['training'] != True].drop(['training'], axis=1)
