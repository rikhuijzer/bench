from core.training_data import *


def test_tokenizer():
    tokenized = 'How to partially upgrade Ubuntu 11 . 10 from Ubuntu 11 . 04 ? '
    detokenized = 'How to partially upgrade Ubuntu 11.10 from Ubuntu 11.04?'
    assert tokenized == tokenize(detokenized)
    assert detokenized == tokenize(tokenized, detokenize=True)


def test_convert_index():
    def helper(text: str, token_index: int, expected: int, begin: bool):
        if begin:
            index = convert_index_begin(text, token_index)
        else:
            index = convert_index_end(text, token_index)
        assert expected == index

    text = 'Upgrading from 11.10 to 12.04'
    helper(text, 6, 24, begin=True)
    helper(text, 8, 29, begin=False)


def test_find_space():
    def helper(text: str, n: int, char_index: int, char: str):
        result = find_space(text, n)
        assert char_index == result
        if char == '':  # end of line, so text[result] would raise index out of bounds error
            assert char_index == len(text)
        else:
            assert char == text[result]

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


def test_fix_tokenisation_index():
    text = 'Upgrading from 11.10 to 12.04'
    tokenized = 'Upgrading from 11 . 10 to 12 . 04'
    start_tokenized = 26
    start_expected = 24
    assert start_expected == fix_tokenisation_index(tokenized, start_tokenized)


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
