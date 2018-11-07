import enum
import typing

import rasa_nlu.training_data

Focus = enum.Enum('Focus', 'ALL INTENT')

Header = enum.Enum('Header', 'JSON YML')

Corpus = enum.Enum('Corpus', 'ASKUBUNTU CHATBOT WEBAPPLICATIONS SNIPS MOCK EMPTY')

CSVs = enum.Enum('CSVs', 'GENERAL INTENTS ENTITIES')

Sentence = typing.NamedTuple('Sentence', [('text', str), ('corpus', Corpus)])

System = typing.NamedTuple('System', [('name', str), ('knowledge', Corpus), ('data', typing.Tuple)])

SystemCorpus = typing.NamedTuple('SystemCorpus', [('system', System), ('corpus', Corpus)])

Query = typing.NamedTuple('Query', [('system', System), ('text', str)])

Response = typing.NamedTuple('Response', [('intent', str), ('confidence', float), ('entities', typing.List[dict])])

Classification = typing.NamedTuple('IntentClassification', [('system_corpus', SystemCorpus), ('response', Response)])

# Classifications = typing.NamedTuple('IntentClassifications', [('system', System), ('df', pd.DataFrame)])

F1Score = typing.NamedTuple('F1Score', [('system', System), ('score', typing.Tuple[float, ...])])

Messages = typing.Tuple[rasa_nlu.training_data.Message, ...]

CSVIntent = typing.NamedTuple('CSVIntent', [('id', int), ('run', int), ('sentence', str), ('intent', str),
                                            ('classification', str), ('confidence', float), ('time', int)])

CSV = typing.NamedTuple('CSV', [('filename', str), ('named_tuple', type)])
