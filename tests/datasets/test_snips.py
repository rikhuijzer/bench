from src.datasets.snips import get_folders, convert_data_spans, convert_data_text, convert_data_message
from src.dataset import convert_message_to_annotated_str
import src.typ as tp
import typing


def get_test_data() -> typing.List[typing.Dict[str, str]]:
    return [
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
    spans = convert_data_spans(get_test_data())
    print()
    tmp = convert_data_text(get_test_data())
    for span in spans:
        print(span, tmp[span[0]:span[1]])


def test_convert_data_message():
    message = convert_data_message(tp.Corpus.SNIPS2017, 'AddToPlaylist', get_test_data(), training=True)
    expected = 'add [Foo](entity_name) songs in [my](playlist_owner) playlist [música libre](playlist)'
    assert expected == convert_message_to_annotated_str(message)
