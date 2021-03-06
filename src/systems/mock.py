import typing

from rasa_nlu.training_data.message import Message

import src.dataset
import src.typ
import src.typ as tp
import src.utils


def get_timestamp() -> str:
    """Just defining a timestamp variable does not seem to always work."""
    return '2018-11-03 16:43:08'


def get_mock_messages(corpus: tp.Corpus) -> typing.Iterable[Message]:
    def create_mock_message(x: int) -> Message:
        return src.dataset.create_message(text=str(x), intent='A' if 0 <= x < 10 else 'B', entities=[],
                                          training=True if x < 15 else False, corpus=src.typ.Corpus.MOCK)

    return map(create_mock_message, range(0, 20))


def train(system_corpus: src.typ.SystemCorpus) -> src.typ.System:
    data = list(system_corpus.system.data)
    data[0] += 1
    return src.typ.System(system_corpus.system.name, system_corpus.corpus, get_timestamp(), tuple(data))


def get_response(query: src.typ.Query) -> src.typ.Response:
    value = int(query.text)
    classification = 'C' if (value % query.system.data[0] == 0) else ('A' if (0 < value <= 10) else 'B')
    return src.typ.Response(classification, 1.0, [])
