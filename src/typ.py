from enum import Enum
from typing import Tuple, NamedTuple, TypeVar, Union, List
from rasa_nlu.training_data.message import Message

Focus = Enum('Focus', 'ALL INTENT')

Header = Enum('Header', 'JSON YML')

Corpus = Enum('Corpus', 'ASKUBUNTU CHATBOT WEBAPPLICATIONS SNIPS2017 MOCK EMPTY')

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
