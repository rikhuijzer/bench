import json
import typing

from rasa_nlu.training_data.message import Message

import src.typ as tp
import src.utils
from src.dataset import get_path, convert_nlu_evaluation_entity, create_message


def read_nlu_evaluation_corpora(corpus: tp.Corpus) -> typing.Tuple[Message, ...]:
    """ Convert NLU Evaluation Corpora dictionary to the internal representation. """
    file = src.utils.get_root() / 'datasets' / get_path(corpus)
    with open(str(file), 'rb') as f:
        js = json.load(f)

    def convert_entities(sentence: dict) -> typing.List[dict]:
        return list(map(lambda e: convert_nlu_evaluation_entity(sentence['text'], e), sentence['entities']))

    def convert_sentence(sentence: dict) -> Message:
        return create_message(sentence['text'], sentence['intent'], convert_entities(sentence),
                              sentence['training'], corpus)

    return tuple(map(convert_sentence, js['sentences']))
