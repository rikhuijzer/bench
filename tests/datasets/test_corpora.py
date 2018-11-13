from src.dataset import ( create_message, convert_message_to_annotated_str, get_messages )
from src.datasets.corpora import convert_index, convert_nlu_evaluation_entity
import typing
import src.typ as tp


def test_convert_index():
    def helper(text: str, token_index: int, expected: int, start: bool):
        index = convert_index(text, token_index, start)
        assert expected == index

    sentence = 'Upgrading from 11.10 to 12.04'
    helper(sentence, 6, 24, start=True)
    helper(sentence, 8, 29, start=False)


def test_nlu_evaluation_entity_converter():
    def helper(text: str, entity: dict, expected: str):
        result = convert_nlu_evaluation_entity(text, entity)
        message = create_message(text, 'some intent', [result], False, tp.Corpus.MOCK)
        assert expected == convert_message_to_annotated_str(message)

    helper(text='when is the next train in muncher freiheit?',
           entity={'entity': 'Vehicle', 'start': 4, 'stop': 4, 'text': 'train'},
           expected='when is the next [train](Vehicle) in muncher freiheit?')

    helper(text='Upgrading from 11.10 to 12.04',
           entity={"text": "12.04", "entity": "UbuntuVersion", "stop": 8, "start": 6},
           expected='Upgrading from 11.10 to [12.04](UbuntuVersion)')

    helper(text='Archive/export all the blog entries from a RSS feed in Google Reader',
           entity={"text": "Google Reader", "entity": "WebService", "stop": 13, "start": 12},
           expected='Archive/export all the blog entries from a RSS feed in [Google Reader](WebService)')


# NLU-Evaluation-Corpora expected_length provided at https://github.com/sebischair/NLU-Evaluation-Corpora
def test_get_messages():
    """ Test whether all corpora get imported correctly.
            All crammed in one function, to avoid having many errors when one of the sub-functions fails.
    """
    def helper(messages: typing.Tuple, expected_length: int, first_row: dict, last_row: dict):
        assert expected_length == len(messages)
        assert first_row == messages[0].as_dict()
        assert last_row == messages[-1].as_dict()

    sentences = get_messages(tp.Corpus.ASKUBUNTU)
    first_row = {
        'text': 'What software can I use to view epub documents?',
        'intent': 'Software Recommendation',
        'training': False,
        'corpus': tp.Corpus.ASKUBUNTU
    }
    last_row = {
        'text': 'What graphical utility can I use for Ubuntu auto shutdown?',
        'intent': 'Shutdown Computer',
        'training': True,
        'corpus': tp.Corpus.ASKUBUNTU
    }

    helper(sentences, 162, first_row, last_row)

    sentences = get_messages(tp.Corpus.CHATBOT)
    first_row = {
        'entities': [{'end': 24,
                      'entity': 'StationDest',
                      'start': 13,
                      'value': 'marienplatz'}],
        'intent': 'FindConnection',
        'text': 'i want to go marienplatz',
        'training': False,
        'corpus': tp.Corpus.CHATBOT
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
        'training': True,
        'corpus': tp.Corpus.CHATBOT
    }
    helper(sentences, 206, first_row, last_row)

    sentences = get_messages(tp.Corpus.WEBAPPLICATIONS)
    first_row = {
        'entities': [{'end': 23,
                      'entity': 'WebService',
                      'start': 15,
                      'value': 'Facebook'}],
        'intent': 'Find Alternative',
        'text': 'Alternative to Facebook',
        'training': False,
        'corpus': tp.Corpus.WEBAPPLICATIONS
    }
    last_row = {
        'entities': [{'end': 31,
                      'entity': 'WebService',
                      'start': 24,
                      'value': 'Harvest'}],
        'intent': 'Delete Account',
        'text': 'How to disable/delete a Harvest account?',
        'training': True,
        'corpus': tp.Corpus.WEBAPPLICATIONS
    }

    helper(sentences, 89, first_row, last_row)
