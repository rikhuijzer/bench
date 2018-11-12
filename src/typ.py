import enum
import typing
from typing import Tuple, NamedTuple

from rasa_nlu.training_data.message import Message

Focus = enum.Enum('Focus', 'ALL INTENT')

Header = enum.Enum('Header', 'JSON YML')

Corpus = enum.Enum('Corpus', 'ASKUBUNTU CHATBOT WEBAPPLICATIONS SNIPS2017 MOCK EMPTY')

CSVs = enum.Enum('CSVs', 'GENERAL INTENTS ENTITIES')

Run = enum.Enum('Run', 'PREVIOUS ALL NEW')

# Messages = Tuple[Message, ...]

# Sentence = NamedTuple('Sentence', [('text', str), ('corpus', Corpus)])  # this one is replaced by field in message

System = NamedTuple('System', [('name', str), ('knowledge', Corpus), ('timestamp', str), ('data', Tuple)])

SystemCorpus = NamedTuple('SystemCorpus', [('system', System), ('corpus', Corpus)])

Query = NamedTuple('Query', [('system', System), ('text', str)])

Response = NamedTuple('Response', [('intent', str), ('confidence', float), ('entities', typing.List[dict])])

Classification = NamedTuple('Classification', [('system', System), ('message', Message), ('response', Response)])

F1Score = NamedTuple('F1Score', [('system', System), ('score', Tuple[float, ...])])

CSVGeneral = NamedTuple('CSVGeneral', [])

CSVIntent = NamedTuple('CSVIntent', [('id', int), ('timestamp', str), ('sentence', str), ('intent', str),
                                     ('classification', str), ('confidence', float), ('time', int)])

CSVEntity = NamedTuple('CSVEntity', [])

CSV_types = typing.Union[CSVGeneral, CSVIntent, CSVEntity]

CSV = NamedTuple('CSV', [('filename', str), ('named_tuple', type)])
