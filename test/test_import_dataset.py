import unittest

from core import import_dataset
from core.import_dataset import *
from core.import_dataset import Corpus


class TestImportDataset(unittest.TestCase):
    dummy_corpus = [
        Sentence('lorem', 'foo', [], True),
        Sentence('ipsum', 'bar', [], False),
        Sentence('dolor', 'baz', [], False)
    ]

    def _validate_import(self, sentences: List[Sentence], expected_length: int, first_row: dict, last_row: dict):
        df = sentences_converter(sentences)
        df_length = df.shape[0]
        self.assertEqual(expected_length, df_length)
        self.assertEqual(first_row, df.loc[0].to_dict())
        self.assertEqual(last_row, df.loc[df_length - 1].to_dict())

    # NLU-Evaluation-Corpora details provided at https://github.com/sebischair/NLU-Evaluation-Corpora
    def test__get_corpus_ask_ubuntu(self):
        sentences = import_dataset._get_corpus(Corpus.AskUbuntu)

        first_row = {
            'sentence': 'What software can I use to view epub documents?',
            'intent': 'Software Recommendation',
            'training': False
        }

        last_row = {
            'sentence': 'What graphical utility can I use for Ubuntu auto shutdown?',
            'intent': 'Shutdown Computer',
            'training': True
        }

        self._validate_import(sentences, 162, first_row, last_row)

    def test__get_corpus_chatbot(self):
        sentences = import_dataset._get_corpus(Corpus.Chatbot)

        first_row = {
            'sentence': 'i want to go [marienplatz](StationDest)',
            'intent': 'FindConnection',
            'training': False
        }

        last_row = {
            'sentence': 'from [garching](StationStart) to [studentenstadt](StationDest)',
            'intent': 'FindConnection',
            'training': True
        }

        self._validate_import(sentences, 206, first_row, last_row)

    def test__get_corpus_webapplications(self):
        sentences = import_dataset._get_corpus(Corpus.WebApplications)

        first_row = {
            'sentence': 'Alternative to [Facebook](WebService)',
            'intent': 'Find Alternative',
            'training': False
        }

        last_row = {
            'sentence': 'How to disable/delete a Harvest [account](WebService)?',
            'intent': 'Delete Account',
            'training': True
        }

        self._validate_import(sentences, 89, first_row, last_row)

    def test__get_train(self):
        train = import_dataset.get_train(self.dummy_corpus)
        row = {'sentence': 'lorem', 'intent': 'foo', 'training': True}
        self._validate_import(train, 1, row, row)

    def test__get_test(self):
        test = import_dataset.get_test(self.dummy_corpus)
        first_row = {'sentence': 'ipsum', 'intent': 'bar', 'training': False}
        last_row = {'sentence': 'dolor', 'intent': 'baz', 'training': False}
        self._validate_import(test, 2, first_row, last_row)

    def test_find_nth(self):
        def helper(text: str, regex: re, n: int, char_index: int, char: str = None):
            result = import_dataset.find_nth(text, regex, n)
            self.assertEqual(char_index, result)
            if char:
                self.assertEqual(char, text[result])
            else:  # text[result] is expected to throw string index ouf of range when char == None
                self.assertEqual(char_index, len(text))

        sentence = 'lorem ipsum dolor sit amet'
        helper(sentence, r'\W', 0, 5, ' ')
        helper(sentence, r'\W', 1, 11, ' ')
        helper(sentence, r'\W', 2, 17, ' ')

        sentence = 'Weather tomorrow?'
        helper(sentence, r'\Z|\W', 1, 16, '?')

        sentence = 'Weather tomorrow'
        helper(sentence, r'\Z|\W', 1, 16, None)

        sentence = 'Weather tomorrow morning?'
        helper(sentence, r'\Z|\W', 1, 16, ' ')

    def test_luis_tokenizer(self):
        sentence = 'How to partially upgrade Ubuntu 11.10 from Ubuntu 11.04?'
        expected = 'How to partially upgrade Ubuntu 11 . 10 from Ubuntu 11 . 04 ? '
        self.assertEqual(expected, tokenizer(sentence))

    def test__str__(self):
        text = 'Could I pay you 50 yen tomorrow or tomorrow?'
        expected = 'Could I pay you 50 [yen](currency lorem ipsum) [tomorrow](date) or [tomorrow](date)?'
        entities = [
            Entity('currency lorem ipsum', 19, 22),
            Entity('date', 23, 31),
            Entity('date', 35, 43)
        ]
        sentence = Sentence(text, 'foo', entities)
        self.assertEqual(expected, str(sentence))

    def test_nlu_evaluation_entity_converter(self):
        text = 'when is the next train in muncher freiheit?'
        # note that entity start and stop look at word, not character
        entity = {'entity': 'Vehicle', 'start': 4, 'stop': 4, 'text': 'train'}
        entity_vehicle = Entity('Vehicle', 17, 22)
        result = import_dataset._nlu_evaluation_entity_converter(text, entity)
        self.assertEqual(str(entity_vehicle), str(result))

        result = Sentence(text, 'intent', [entity_vehicle])
        expected = 'when is the next [train](Vehicle) in muncher freiheit?'
        self.assertEqual(expected, str(result))


if __name__ == '__main__':
    unittest.main()
