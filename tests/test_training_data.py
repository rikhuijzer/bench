import core.training_data
import core.typ
import typing
import functools


def test_convert_index():
    def helper(text: str, token_index: int, expected: int, start: bool):
        index = core.training_data.convert_index(text, token_index, start)
        assert expected == index

    text = 'Upgrading from 11.10 to 12.04'
    helper(text, 6, 24, start=True)
    helper(text, 8, 29, start=False)


def test_message_to_annotated_str():
    text = 'Could I pay you 50 yen tomorrow or tomorrow?'
    expected = 'Could I pay you 50 [yen](currency lorem ipsum) [tomorrow](date) or [tomorrow](date)?'
    entities = [
        core.training_data.create_entity(19, 22, 'currency lorem ipsum', 'yen'),
        core.training_data.create_entity(23, 31, 'date', 'tomorrow'),
        core.training_data.create_entity(35, 43, 'date', 'tomorrow')
    ]
    message = core.training_data.create_message(text, 'foo', entities, False)

    assert expected, core.training_data.convert_message_to_annotated_str(message)


def test_nlu_evaluation_entity_converter():
    def helper(text: str, entity: dict, expected: str):
        result = core.training_data.convert_nlu_evaluation_entity(text, entity)
        message = core.training_data.create_message(text, 'some intent', [result], False)
        assert expected == core.training_data.convert_message_to_annotated_str(message)

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

    sentences = core.training_data.get_messages(core.typ.Corpus.ASKUBUNTU)
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

    helper(sentences, 162, first_row, last_row)

    sentences = core.training_data.get_messages(core.typ.Corpus.CHATBOT)
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
    helper(sentences, 206, first_row, last_row)

    sentences = core.training_data.get_messages(core.typ.Corpus.WEBAPPLICATIONS)
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
        'entities': [{'end': 31,
                      'entity': 'WebService',
                      'start': 24,
                      'value': 'Harvest'}],
        'intent': 'Delete Account',
        'text': 'How to disable/delete a Harvest account?',
        'training': True
    }

    helper(sentences, 89, first_row, last_row)


def test_get_intents():
    expected = {'Delete Account', 'Find Alternative', 'Download Video',
                'Filter Spam', 'Change Password', 'Sync Accounts', 'None', 'Export Data'}
    assert expected == set(core.training_data.get_intents(core.typ.Corpus.WEBAPPLICATIONS))


def test_get_filtered_messages():
    func = functools.partial(core.training_data.get_filtered_messages, corpus=core.typ.Corpus.MOCK)
    assert 15 == len(tuple(func(train=True)))
    assert 5 == len(tuple(func(train=False)))
