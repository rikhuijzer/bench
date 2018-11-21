import csv
from pathlib import Path
from typing import Tuple, Optional, List

from rasa_nlu.training_data.message import Message
from sklearn.model_selection import train_test_split

import src.typ as tp
from src.dataset import get_messages
from src.utils import get_root


def convert_message_line(message: tp.Message, train) -> Optional[Tuple]:
    if message.data['training'] == train:
        return message.text, message.data['intent']


def write_tsv(messages: Tuple[Message, ...], filename: Path, train: bool):
    with open(str(filename), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')

        for message in messages:
            line = convert_message_line(message, train)
            if line:
                tsv_writer.writerow(line)


def get_X_y(messages: Tuple[Message, ...]) -> Tuple[Tuple, Tuple]:
    X = tuple(map(lambda message: message.text, messages))
    y = tuple(map(lambda message: message.data['intent'], messages))
    return X, y


def get_split(X: List, y: List) -> tp.Split:
    """Return train / dev / test split."""

    def my_train_test_split(my_X, my_y):
        return train_test_split(my_X, my_y, test_size=0.2, random_state=0, stratify=my_y)

    X_train, X_dev, y_train, y_dev = my_train_test_split(X, y)
    X_train, X_test, y_train, y_test = my_train_test_split(X_train, y_train)

    train = tp.Xy(X_train, y_train)
    dev = tp.Xy(X_dev, y_dev)
    test = tp.Xy(X_test, y_test)
    return tp.Split(train, dev, test)


def to_tsv(corpus: tp.Corpus, folder: Path):
    messages = get_messages(corpus)
    write_tsv(messages, folder / 'train.tsv', train=True)
    write_tsv(messages, folder / 'test.tsv', train=False)


if __name__ == '__main__':
    to_tsv(tp.Corpus.ASKUBUNTU, get_root() / 'generated' / 'askubuntu')
