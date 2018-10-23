import unittest

from core import import_dataset
from core.import_dataset import *
from core.import_dataset import Corpus


class TestImportDataset(unittest.TestCase):
    dummy_corpus = pd.DataFrame({
        'sentence': ['lorem', 'ipsum', 'dolor'],
        'intent': ['foo', 'bar', 'baz'],
        'training': [True, False, False]
    })

    def _validate_import(self, df: pd.DataFrame, expected_length: int, first_row: dict, last_row: dict):
        df_length = df.shape[0]
        self.assertEqual(expected_length, df_length)
        self.assertEqual(first_row, df.loc[0].to_dict())
        self.assertEqual(last_row, df.loc[df_length - 1].to_dict())

    # NLU-Evaluation-Corpora details provided at https://github.com/sebischair/NLU-Evaluation-Corpora
    def test__get_corpus_ask_ubuntu(self):
        df = import_dataset._get_corpus(Corpus.AskUbuntu)

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

        self._validate_import(df, 162, first_row, last_row)

    def test__get_corpus_chatbot(self):
        df = import_dataset._get_corpus(Corpus.Chatbot)

        first_row = {
            'sentence': 'i want to go marienplatz',
            'intent': 'FindConnection',
            'training': False
        }

        last_row = {
            'sentence': 'from garching to studentenstadt',
            'intent': 'FindConnection',
            'training': True
        }

        self._validate_import(df, 206, first_row, last_row)

    def test__get_corpus_webapplications(self):
        df = import_dataset._get_corpus(Corpus.WebApplications)

        first_row = {
            'sentence': 'Alternative to Facebook',
            'intent': 'Find Alternative',
            'training': False
        }

        last_row = {
            'sentence': 'How to disable/delete a Harvest account?',
            'intent': 'Delete Account',
            'training': True
        }

        self._validate_import(df, 89, first_row, last_row)

    def test__get_train(self):
        train = import_dataset._get_train(pd.DataFrame(self.dummy_corpus))
        row = {'sentence': 'lorem', 'intent': 'foo'}
        self._validate_import(train, 1, row, row)

    def test__get_test(self):
        test = import_dataset._get_test(self.dummy_corpus)
        first_row = {'sentence': 'ipsum', 'intent': 'bar'}
        last_row = {'sentence': 'dolor', 'intent': 'baz'}
        self._validate_import(test, 2, first_row, last_row)

    def test_find_nth(self):
        sentence = 'lorem ipsum dolor sit amet'
        self.assertEqual(5, import_dataset.find_nth(sentence, r' ', 0))
        self.assertEqual(11, import_dataset.find_nth(sentence, r' ', 1))
        self.assertEqual(17, import_dataset.find_nth(sentence, r' ', 2))

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
        entity = {'entity': 'Vehicle', 'start': 4, 'stop': 4, 'text': 'train'}
        expected = Entity('Vehicle', 16, 22)
        result = import_dataset._nlu_evaluation_entity_converter(text, entity)
        # self.assertEqual(str(expected), str(result))

        entity = {'entity': 'Foo', 'start': 6, 'stop': 7, 'text': 'muncher feiheit'}
        expected = Entity('Foo', 25, 41)
        result = import_dataset._nlu_evaluation_entity_converter(text, entity)
        # self.assertEqual(str(expected), str(result))


if __name__ == '__main__':
    unittest.main()
