import unittest

from core import training_data
from core.training_data import *


class TestTrainingData(unittest.TestCase):
    dummy_corpus = [
        create_message('lorem', 'foo', [], True),
        create_message('ipsum', 'bar', [], False),
        create_message('dolor', 'baz', [], False)
    ]

    def test_nlu_evaluation_entity_converter(self):
        text = 'when is the next train in muncher freiheit?'
        # note that entity start and stop look at word, not character
        entity = {'entity': 'Vehicle', 'start': 4, 'stop': 4, 'text': 'train'}
        result = training_data._nlu_evaluation_entity_converter(text, entity)
        message = create_message(text, 'some intent', [result], False)
        expected = 'when is the next [train](Vehicle) in muncher freiheit?'
        self.assertEqual(expected, message_to_annotated_str(message))

    def test_find_space(self):
        def helper(text: str, n: int, char_index: int, char: str):
            result = training_data.find_space(text, n)
            self.assertEqual(char_index, result)
            if char == '':  # end of line, so text[result] would raise index out of bounds error
                self.assertEqual(char_index, len(text))
            else:
                self.assertEqual(char, text[result])

        sentence = 'lorem ipsum dolor sit amet'
        helper(sentence, 1, 5, ' ')
        helper(sentence, 2, 11, ' ')
        helper(sentence, 3, 17, ' ')
        helper(sentence, 5, len(sentence), '')

        sentence = 'Weather tomorrow?'
        helper(sentence, 2, 16, '?')

        sentence = 'Weather tomorrow'
        helper(sentence, 2, 16, '')

        sentence = 'Weather tomorrow morning?'
        helper(sentence, 2, 16, ' ')

    def test_tokenizer(self):
        tokenized = 'How to partially upgrade Ubuntu 11 . 10 from Ubuntu 11 . 04 ? '
        detokenized = 'How to partially upgrade Ubuntu 11.10 from Ubuntu 11.04?'
        self.assertEqual(tokenized, tokenize(detokenized))
        self.assertEqual(detokenized, tokenize(tokenized, detokenize=True))

    def test_message_to_annotated_str(self):
        text = 'Could I pay you 50 yen tomorrow or tomorrow?'
        expected = 'Could I pay you 50 [yen](currency lorem ipsum) [tomorrow](date) or [tomorrow](date)?'
        entities = [
            create_entity(19, 22, 'currency lorem ipsum', 'yen'),
            create_entity(23, 31, 'date', 'tomorrow'),
            create_entity(35, 43, 'date', 'tomorrow')
        ]
        message = create_message(text, 'foo', entities, False)
        self.assertEqual(expected, message_to_annotated_str(message))

    def helper_get_corpus(self, messages: List[Message], expected_length: int, first_row: dict, last_row: dict):
        self.assertEqual(expected_length, len(messages))
        self.assertEqual(first_row, messages[0].as_dict())
        self.assertEqual(last_row, messages[-1].as_dict())

    def test_get_train(self):
        train = training_data.get_train(self.dummy_corpus)
        row = {'text': 'lorem', 'intent': 'foo', 'training': True}
        self.helper_get_corpus(train, 1, row, row)

    def test_get_test(self):
        test = training_data.get_test(self.dummy_corpus)
        first_row = {'text': 'ipsum', 'intent': 'bar', 'training': False}
        last_row = {'text': 'dolor', 'intent': 'baz', 'training': False}

        self.helper_get_corpus(test, 2, first_row, last_row)

    # NLU-Evaluation-Corpora expected_length provided at https://github.com/sebischair/NLU-Evaluation-Corpora
    def test_get_corpus(self):
        """ Test whether all corpora get imported correctly.
        All crammed in one function, to avoid having many errors when one of the sub-functions (like find-nth) fails.
        """
        sentences = get_corpus(Corpus.AskUbuntu)
        first_row = {
            'text': 'What software can I use to view epub documents?',
            'intent': 'Software Recommendation',
            'training': False
        }
        last_row = {
            'text': 'What graphical utility can I use for Ubuntu auto shutdown?',
            'intent': 'Shutdown Computer',
            'training': True
        }
        self.helper_get_corpus(sentences, 162, first_row, last_row)

        sentences = get_corpus(Corpus.Chatbot)
        first_row = {
            'entities': [{'end': 24,
                          'entity': 'StationDest',
                          'start': 13,
                          'value': 'marienplatz'}],
            'intent': 'FindConnection',
            'text': 'i want to go marienplatz',
            'training': False
        }
        last_row = {
            'entities': [{'end': 13,
                          'entity': 'StationStart',
                          'start': 5,
                          'value': 'garching'},
                         {'end': 31,
                          'entity': 'StationDest',
                          'start': 17,
                          'value': 'studentenstadt'}],
            'intent': 'FindConnection',
            'text': 'from garching to studentenstadt',
            'training': True
        }
        self.helper_get_corpus(sentences, 206, first_row, last_row)

        sentences = get_corpus(Corpus.WebApplications)
        first_row = {
            'entities': [{'end': 23,
                          'entity': 'WebService',
                          'start': 15,
                          'value': 'Facebook'}],
            'intent': 'Find Alternative',
            'text': 'Alternative to Facebook',
            'training': False
        }
        last_row = {
            'entities': [{'end': 39,
                          'entity': 'WebService',
                          'start': 32,
                          'value': 'Harvest'}],
            'intent': 'Delete Account',
            'text': 'How to disable/delete a Harvest account?',
            'training': True
        }
        self.helper_get_corpus(sentences, 89, first_row, last_row)


if __name__ == '__main__':
    unittest.main()
