import csv
from pathlib import Path
from typing import Tuple, Optional, List, Iterable

from rasa_nlu.training_data.message import Message
from sklearn.model_selection import train_test_split

import src.typ as tp
from src.dataset import get_messages
from src.utils import get_root


def convert_message_line(message: tp.Message, train) -> Optional[Tuple]:
    if message.data['training'] == train:
        return message.text, message.data['intent']


def write_tsv(Xys: List[tp.Xy], filename: Path):
    with open(str(filename), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for X, y in zip(Xys.X, Xys.y):
            tsv_writer.writerow([X, y])


def get_X_y(messages: Tuple[Message, ...]) -> Tuple[List, List]:
    X = list(map(lambda message: message.text, messages))
    y = list(map(lambda message: message.data['intent'], messages))
    return X, y


def get_split(X: List, y: List) -> tp.Split:
    """Return stratified train / dev / test split."""

    def my_train_test_split(my_X, my_y, split_size: float):
        return train_test_split(my_X, my_y, test_size=split_size, random_state=0, stratify=my_y)

    X_train, X_dev, y_train, y_dev = my_train_test_split(X, y, split_size=0.20)
    X_train, X_test, y_train, y_test = my_train_test_split(X_train, y_train, split_size=0.25)

    train = tp.Xy(X_train, y_train)
    dev = tp.Xy(X_dev, y_dev)
    test = tp.Xy(X_test, y_test)
    return tp.Split(train, dev, test)


def to_tsv(corpus: tp.Corpus, folder: Path):
    messages = get_messages(corpus)
    split = get_split(*get_X_y(messages))
    write_tsv(split.train, folder / 'train.tsv')
    write_tsv(split.dev, folder / 'dev.tsv')
    write_tsv(split.test, folder / 'test.tsv')


if __name__ == '__main__':
    to_tsv(tp.Corpus.ASKUBUNTU, get_root() / 'generated' / 'askubuntu')
