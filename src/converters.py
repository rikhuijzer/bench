from src.dataset import get_messages
import src.typ as tp
from pathlib import Path
from src.utils import get_root
from typing import Tuple, Optional, List
from rasa_nlu.training_data.message import Message
import csv


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


def to_tsv(corpus: tp.Corpus, folder: Path):
    messages = get_messages(corpus)
    write_tsv(messages, folder / 'train.tsv', train=True)
    write_tsv(messages, folder / 'test.tsv', train=False)


if __name__ == '__main__':
    to_tsv(tp.Corpus.ASKUBUNTU, get_root() / 'generated' / 'askubuntu')
