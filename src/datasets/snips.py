import typing
from rasa_nlu.training_data.message import Message
import src.typ as tp
from src import dataset
import pathlib
from typing import List, Iterable
from itertools import accumulate, chain
import operator
from functools import reduce


def get_folders(corpus: tp.Corpus) -> Iterable[pathlib.Path]:
    """Get all folders listed in some corpus folder."""
    return filter(lambda f: f.is_dir(), dataset.get_path(corpus).glob('./*'))


def convert_data_text(data: List[dict]) -> str:
    return ''.join([item['text'] for item in data])


def convert_data_spans(data: List[dict]) -> Iterable[typing.Tuple[int, int]]:
    """Get start and end index for part of a sentence, see SNIPS 2017 .json files for examples."""
    lengths = list(map(lambda item: len(item['text']), data))
    start_indexes = [0] + list(accumulate(lengths, operator.add))[:-1]
    end_indexes = map(lambda x: x[0] + x[1], zip(start_indexes, lengths))
    return zip(start_indexes, end_indexes)


def convert_data_entities(data: List[dict]) -> Iterable[dict]:
    """Returns entities in Rasa representation for some SNIPS data element."""
    spans = convert_data_spans(data)
    for span, item in zip(spans, data):
        if 'entity' in item:
            yield dataset.create_entity(start=span[0], end=span[1], entity=item['entity'], value=item['text'])


def convert_data_message(corpus: tp.Corpus, intent: str, data: List[dict], training: bool) -> Message:
    """Returns message in Rasa representation for some SNIPS data element."""
    text = convert_data_text(data)
    entities = list(convert_data_entities(data))
    return dataset.create_message(text, intent, entities, training, corpus)


def convert_file_messages(corpus: tp.Corpus, file: pathlib.Path, intent: str, training: bool) -> Iterable[Message]:
    """Returns messages in Rasa representation for some SNIPS .json file."""
    js = dataset.convert_json_dict(file)
    return map(lambda item: convert_data_message(corpus, intent, item['data'], training), js[intent])


def read_snips2017(corpus: tp.Corpus) -> Iterable[Message]:
    def get_messages(folder: pathlib.Path) -> Iterable[Message]:
        intent = folder.name
        filename = folder / 'train_{}.json'.format(intent)
        return convert_file_messages(corpus, filename, intent, training=True)

    folders = get_folders(corpus)
    nested_messages = map(lambda folder: get_messages(folder), folders)
    return chain.from_iterable(nested_messages)
