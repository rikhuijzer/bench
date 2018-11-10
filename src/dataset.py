import functools
import pathlib
import typing

import rasa_nlu.training_data
import rasa_nlu.training_data
import rasa_nlu.training_data.formats.markdown
import rasa_nlu.training_data.formats.markdown
import rasa_nlu.utils
import rasa_nlu.utils
from rasa_nlu.training_data import Message

import src.systems.mock
import src.systems.mock
import src.typ as tp
import src.utils
import src.utils
from src.datasets.nlu_evaluation_corpora import read_nlu_evaluation_corpora
from src.datasets.snips import read_snips2017
import pandas as pd
import json


def convert_json_dict(file: pathlib.Path) -> dict:
    with open(str(file), 'rb') as f:
        return json.load(f)


def get_path(corpus: tp.Corpus) -> pathlib.Path:
    if corpus == tp.Corpus.MOCK or corpus == tp.Corpus.EMPTY:
        raise AssertionError('This function should not be called on {}.'.format(corpus))
    paths = {
        tp.Corpus.ASKUBUNTU: pathlib.Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json',
        tp.Corpus.CHATBOT: pathlib.Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json',
        tp.Corpus.WEBAPPLICATIONS: pathlib.Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json',
        tp.Corpus.SNIPS2017: pathlib.Path('2017-06-custom-intent-engines')
    }
    return src.utils.get_root() / 'datasets' / paths[corpus]


def create_entity(start: int, end: int, entity: str, value: str) -> dict:
    return {'start': start, 'end': end, 'entity': entity, 'value': value}


def create_message(text: str, intent: str, entities: [], training: bool, corpus: tp.Corpus) -> Message:
    """ Helper function to create a message: Message used by Rasa including whether train or test sentence. """
    message = Message.build(text, intent, entities)
    message.data['training'] = training
    message.data['corpus'] = corpus
    return message


def convert_message_to_annotated_str(message: Message) -> str:
    """ Convert Message object to string having annotated entities. """
    training_examples: typing.List[Message] = [message]
    training_data = rasa_nlu.training_data.TrainingData(training_examples=training_examples)
    generated = rasa_nlu.training_data.formats.markdown.MarkdownWriter()._generate_training_examples_md(training_data)
    generated = generated[generated.find('\n') + 3:-1]  # removes header
    return generated


def messages_to_dataframe(messages: typing.Iterable[Message], focus=tp.Focus.ALL) -> pd.DataFrame:
    """ Returns a DataFrame (table) from a list of Message objects which can be used for visualisation.

    Args:
        messages: Sentences. Sentence is annotated if (focus != 'intent')
        focus: Focus of the DataFrame.  For intent classification visualisation choose 'intent'
    """
    pd.set_option('max_colwidth', 180)

    data = {'message': [], 'intent': [], 'training': []}
    for message in messages:
        intent_focus = focus == tp.Focus.INTENT
        data['message'].append(message.text if intent_focus else convert_message_to_annotated_str(message))
        data['intent'].append(message.data['intent'])
        data['training'].append(message.data['training'])
    return pd.DataFrame(data)


@functools.lru_cache()
def get_messages(corpus: tp.Corpus) -> typing.Tuple[Message, ...]:
    """ Get all messages: Message from some file containing corpus and cache the messages. """
    functions = {  # tp.Corpus -> typing.Iterable[Message]
        tp.Corpus.MOCK: src.systems.mock.get_mock_messages,
        tp.Corpus.WEBAPPLICATIONS: read_nlu_evaluation_corpora,
        tp.Corpus.CHATBOT: read_nlu_evaluation_corpora,
        tp.Corpus.ASKUBUNTU: read_nlu_evaluation_corpora,
        tp.Corpus.SNIPS2017: read_snips2017
    }
    return tuple(functions[corpus](corpus))


def get_filtered_messages(corpus: tp.Corpus, train: bool) -> typing.Iterable[Message]:
    return filter(lambda m: train == m.data['training'], get_messages(corpus))


def get_intents(corpus: tp.Corpus) -> typing.Iterable[str]:
    """ Returns intent for each message in some corpus. To get unique intents one can simply cast it to a set. """
    return map(lambda m: m.data['intent'], get_messages(corpus))
