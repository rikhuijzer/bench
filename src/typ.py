from enum import Enum, auto
from typing import Tuple, NamedTuple, TypeVar, Union, List
from rasa_nlu.training_data.message import Message

Focus = Enum('Focus', 'ALL INTENT')

Header = Enum('Header', 'JSON YML')


class Corpus(Enum):
    ASKUBUNTU = auto()
    CHATBOT = auto()
    WEBAPPLICATIONS = auto()
    SNIPS2017 = auto()
    MOCK = auto()
    EMPTY = auto()


CSVs = Enum('CSVs', 'STATS INTENTS ENTITIES')

T = TypeVar('T')

System = NamedTuple('System', [('name', str), ('knowledge', Corpus), ('timestamp', str), ('data', Tuple)])

SystemCorpus = NamedTuple('SystemCorpus', [('system', System), ('corpus', Corpus)])

Query = NamedTuple('Query', [('system', System), ('text', str)])

Response = NamedTuple('Response', [('intent', str), ('confidence', float), ('entities', List[dict])])

Classification = NamedTuple('Classification', [('system', System), ('message', Message), ('response', Response)])

F1Score = NamedTuple('F1Score', [('system', System), ('score', Tuple[float, ...])])

CSVStats = NamedTuple('CSVStats', [])

CSVIntent = NamedTuple('CSVIntent', [('id', int), ('timestamp', str), ('sentence', str), ('gold_standard', str),
                                     ('classification', str), ('confidence', float), ('time', int)])

CSVEntity = NamedTuple('CSVEntity', [('id', int), ('intent_id', str), ('timestamp', str), ('source', str),
                                     ('entity', str), ('value', str), ('start', int), ('end', int),
                                     ('confidence', float)])

CSV_types = Union[CSVStats, CSVIntent, CSVEntity]

CSV = NamedTuple('CSV', [('filename', str), ('named_tuple', type)])

Xy = NamedTuple('Xy', [('X', Tuple), ('y', Tuple)])

Split = NamedTuple('Split', [('train', Xy), ('dev', Xy), ('test', Xy)])
