import typing
from rasa_nlu.training_data.message import Message
import src.typ as tp
from src.dataset import get_path
import pathlib


def get_folders(path: pathlib.Path) -> typing.Iterable[pathlib.Path]:
    """Get all folders for some path."""
    print(2)


def read_snips2017(corpus: tp.Corpus) -> typing.Iterable[Message]:
    return ()