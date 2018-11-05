import enum
import pathlib
import typing
import pandas as pd
import rasa_nlu.training_data


# TODO: Replace enums by functional enum calls, they should not be used as mapping.
# TODO: The mapping behaviour can be obtained by creating a mapping function for each Enum. Probably best to create
# on function which takes
class StartEnd(enum.Enum):
    start = 0
    end = 1


Focus = enum.Enum('Focus', 'ALL INTENT')


class Header(enum.Enum):  # TODO: We need this header only in a few places, so create function
    json = {'content-type': 'application/json'}
    yml = {'content-type': 'application/x-yml'}


class Corpus:  # TODO: We keep passing path around, while we only need the path one time in the code, makes no sense
    # if both mock and empty get empty string then the enum will return Mock for both (the highest of the two)
    AskUbuntu = pathlib.Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json'
    Chatbot = pathlib.Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json'
    WebApplications = pathlib.Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json'
    Snips = pathlib.Path('snips') / 'benchmark_data.json'
    Mock = 'mock'
    Empty = 'empty'


Sentence = typing.NamedTuple('Sentence', [('text', str), ('corpus', Corpus)])


System = typing.NamedTuple('System', [('name', str), ('knowledge', Corpus), ('data', typing.Tuple)])


SystemCorpus = typing.NamedTuple('SystemCorpus', [('system', System), ('corpus', Corpus)])


Query = typing.NamedTuple('Query', [('system', System), ('text', str)])


Response = typing.NamedTuple('Response', [('intent', str), ('confidence', float), ('entities', typing.List[dict])])


Classification = typing.NamedTuple('IntentClassification', [('system_corpus', SystemCorpus), ('response', Response)])


Classifications = typing.NamedTuple('IntentClassifications', [('system', System), ('df', pd.DataFrame)])


F1Scores = typing.NamedTuple('F1Scores', [('system', System), ('scores', typing.Tuple[float, ...])])


Messages = typing.Tuple[rasa_nlu.training_data.Message, ...]


CSVIntent = typing.NamedTuple('CSVIntent', [('id', int), ('run', int), ('sentence', str), ('intent', str),
                                            ('classification', str), ('confidence', float), ('time', int)])


CSV = typing.NamedTuple('CSV', [('filename', str), ('named_tuple', type)])


class CSVs(enum.Enum):
    General = CSV('general.yml', float)
    Intents = CSV('intents.csv', CSVIntent)
    Entities = CSV('entities.csv', float)
