from src.datasets.snips import (
    get_folders, convert_data_spans, convert_data_message, convert_file_messages, read_snips2017
)
from src.dataset import convert_message_to_annotated_str, get_path
import src.typ as tp
import typing


corpus = tp.Corpus.SNIPS2017
intent = 'AddToPlaylist'
data = [
        {"text": "add "},
        {"text": "Foo", "entity": "entity_name"},
        {"text": " songs in "},
        {"text": "my", "entity": "playlist_owner"},
        {"text": " playlist "},
        {"text": "música libre", "entity": "playlist"}
    ]


def test_get_folders():
    names = [folder.name for folder in get_folders(tp.Corpus.SNIPS2017)]
    assert 7 == len(names)
    assert 'RateBook' in names


def test_convert_data_spans():
    spans = tuple(convert_data_spans(data))
    assert (29, 41) == spans[5]


def test_convert_data_message():
    message = convert_data_message(corpus, intent, data, training=True)
    expected = 'add [Foo](entity_name) songs in [my](playlist_owner) playlist [música libre](playlist)'
    assert expected == convert_message_to_annotated_str(message)


def test_convert_file_messages():
    file = get_path(corpus) / intent / 'train_AddToPlaylist.json'
    messages = tuple(convert_file_messages(corpus, file, intent, training=True))
    expected = 'Add [BSlade](artist) to [women of k-pop](playlist) playlist'
    assert expected == convert_message_to_annotated_str(messages[4])
    assert 300 == len(messages)


def test_read_snips2017():
    messages = read_snips2017(corpus)
    assert 2100 == len(tuple(messages))
