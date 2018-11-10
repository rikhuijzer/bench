from src.dataset import (
    create_entity, create_message, convert_message_to_annotated_str, get_filtered_messages, get_intents
)
import src.typ
import functools


def test_message_to_annotated_str():
    text = 'Could I pay you 50 yen tomorrow or tomorrow?'
    expected = 'Could I pay you 50 [yen](currency lorem ipsum) [tomorrow](date) or [tomorrow](date)?'
    entities = [
        create_entity(19, 22, 'currency lorem ipsum', 'yen'),
        create_entity(23, 31, 'date', 'tomorrow'),
        create_entity(35, 43, 'date', 'tomorrow')
    ]
    message = create_message(text, 'foo', entities, False, src.typ.Corpus.MOCK)

    assert expected, convert_message_to_annotated_str(message)


def test_get_intents():
    expected = {'Delete Account', 'Find Alternative', 'Download Video',
                'Filter Spam', 'Change Password', 'Sync Accounts', 'None', 'Export Data'}
    assert expected == set(get_intents(src.typ.Corpus.WEBAPPLICATIONS))


def test_get_filtered_messages():
    func = functools.partial(get_filtered_messages, corpus=src.typ.Corpus.MOCK)
    assert 15 == len(tuple(func(train=True)))
    assert 5 == len(tuple(func(train=False)))
