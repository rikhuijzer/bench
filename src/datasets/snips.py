import typing
from rasa_nlu.training_data.message import Message
import src.typ as tp
from src import dataset
import pathlib
from typing import List, Iterable


def get_folders(corpus: tp.Corpus) -> Iterable[pathlib.Path]:
    """Get all folders listed in some corpus folder."""
    return filter(lambda f: f.is_dir(), dataset.get_path(corpus).glob('./*'))


def convert_data_text(data: List[dict]) -> str:
    return ''.join([item['text'] for item in data])


def convert_data_spans(data: List[dict]) -> Iterable[typing.Tuple[int, int]]:
    lengths = [len(item['text']) for item in data]
    start_indexes = []
    for i, item in enumerate(data):
        start_indexes.append(len(''.join([item['text'] for item in data[0:i]])))
    end_indexes = list(map(lambda x: x[0] + x[1], zip(start_indexes, lengths)))
    return zip(start_indexes, end_indexes)


def convert_data_entities(data: List[dict]) -> Iterable[dict]:
    spans = convert_data_spans(data)
    for span, item in zip(spans, data):
        if 'entity' in item:
            yield dataset.create_entity(start=span[0], end=span[1], entity=item['entity'], value=item['text'])


def convert_data_message(corpus: tp.Corpus, intent: str, data: List[dict], training: bool) -> Message:
    text = convert_data_text(data)
    entities = list(convert_data_entities(data))
    return dataset.create_message(text, intent, entities, training, corpus)


def convert_file_messages(corpus: tp.Corpus, file: pathlib.Path) -> Iterable[Message]:
    js = dataset.convert_json_dict(file)

    return [Message]


def read_snips2017(corpus: tp.Corpus) -> Iterable[Message]:
    return ()
