import json
import pandas as pd


class Corpus:
    js: json.load

    def __init__(self, file):
        with open(file, 'r') as f:
            self.js = json.load(f)

    def _get_train_test(self, training):
        train = {'sentence': [], 'intent': []}

        for sentence in self.js['sentences']:
            if sentence['training'] == training:
                train['sentence'].append(sentence['text'])
                train['intent'].append(sentence['intent'])

        return pd.DataFrame(train)

    def get_train(self):
        return self._get_train_test(True)

    def get_test(self):
        return self._get_train_test(False)
