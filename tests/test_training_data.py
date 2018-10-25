from core.training_data import *


def test_convert_index():
    def helper(text: str, token_index: int, expected: int, begin: bool):
        index = convert_index(text, token_index, begin)
        assert expected == index

    text = 'Upgrading from 11.10 to 12.04'
    helper(text, 6, 24, begin=True)
    helper(text, 8, 29, begin=False)


def test_message_to_annotated_str():
    text = 'Could I pay you 50 yen tomorrow or tomorrow?'
    expected = 'Could I pay you 50 [yen](currency lorem ipsum) [tomorrow](date) or [tomorrow](date)?'
    entities = [
        build_entity(19, 22, 'currency lorem ipsum', 'yen'),
        build_entity(23, 31, 'date', 'tomorrow'),
        build_entity(35, 43, 'date', 'tomorrow')
    ]
    message = create_message(text, 'foo', entities, False)

    assert expected, message_to_annotated_str(message)


def test_nlu_evaluation_entity_converter():
    def helper(text: str, entity: dict, expected: str):
        result = nlu_evaluation_entity_converter(text, entity)
        message = create_message(text, 'some intent', [result], False)
        assert expected == message_to_annotated_str(message)

    helper(text='when is the next train in muncher freiheit?',
           entity={'entity': 'Vehicle', 'start': 4, 'stop': 4, 'text': 'train'},
           expected='when is the next [train](Vehicle) in muncher freiheit?')

    helper(text='Upgrading from 11.10 to 12.04',
           entity={"text": "12.04", "entity": "UbuntuVersion", "stop": 8, "start": 6},
           expected='Upgrading from 11.10 to [12.04](UbuntuVersion)')
