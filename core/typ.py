import enum
import pathlib
import typing
import pandas as pd
import rasa_nlu.training_data


class StartEnd(enum.Enum):
    start = 0
    end = 1


class Focus(enum.Enum):
    all = 'all'
    intent = 'intent'


class Header(enum.Enum):
    json = {'content-type': 'application/json'}
    yml = {'content-type': 'application/x-yml'}


class Corpus(enum.Enum):
    AskUbuntu = pathlib.Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json'
    Chatbot = pathlib.Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json'
    WebApplications = pathlib.Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json'
    Snips = pathlib.Path('snips') / 'benchmark_data.json'
    Mock = ''
    Empty = ''


class Result(enum.Enum):
    General = 'general.yml'
    Intents = 'intents.csv'
    Entities = 'entities.csv'


Sentence = typing.NamedTuple('Sentence', [('text', str), ('corpus', Corpus)])


System = typing.NamedTuple('System', [('name', str), ('knowledge', Corpus), ('data', typing.Tuple)])


Query = typing.NamedTuple('Query', [('system', System), ('text', str)])


Response = typing.NamedTuple('Response', [('intent', str), ('confidence', float), ('entities', typing.List[dict])])


Classification = typing.NamedTuple('IntentClassification', [('system', System), ('response', Response)])


Classifications = typing.NamedTuple('IntentClassifications', [('system', System), ('df', pd.DataFrame)])


F1Scores = typing.NamedTuple('F1Scores', [('system', System), ('scores', typing.Tuple[float, ...])])


Messages = typing.Tuple[rasa_nlu.training_data.Message, ...]


CSVIntent = typing.NamedTuple('CSVIntent', [('id', str), ('run', int), ('sentence', str), ('intent', str),
                                            ('classification', str), ('confidence', float), ('time', int)])
