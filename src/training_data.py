import json
import functools
import typing
import pandas as pd
import nltk.tokenize
import rasa_nlu.training_data
from rasa_nlu.training_data import Message
import rasa_nlu.training_data.formats.markdown
import rasa_nlu.utils
import src.typ
import src.utils
import pathlib
import src.mock

pd.set_option('max_colwidth', 180)


def get_path(corpus: src.typ.Corpus) -> pathlib.Path:
    if corpus == src.typ.Corpus.MOCK or corpus == src.typ.Corpus.EMPTY:
        raise AssertionError('This function should not be called on {}.'.format(corpus))
    mapping = {
        src.typ.Corpus.ASKUBUNTU: pathlib.Path('NLU-Evaluation-Corpora') / 'AskUbuntuCorpus.json',
        src.typ.Corpus.CHATBOT: pathlib.Path('NLU-Evaluation-Corpora') / 'ChatbotCorpus.json',
        src.typ.Corpus.WEBAPPLICATIONS: pathlib.Path('NLU-Evaluation-Corpora') / 'WebApplicationsCorpus.json',
        src.typ.Corpus.SNIPS: pathlib.Path('snips') / 'benchmark_data.json'
    }
    return mapping[corpus]


def create_entity(start: int, end: int, entity: str, value: str) -> dict:
    return {'start': start, 'end': end, 'entity': entity, 'value': value}


def create_message(text: str, intent: str, entities: [], training: bool, corpus: src.typ.Corpus) -> Message:
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


def convert_nlu_evaluation_entity(text: str, entity: dict) -> dict:
    """ Convert a NLU Evaluation Corpora sentence to Entity object. See test for examples. """
    start = convert_index(text, entity['start'], start=True)
    end = convert_index(text, entity['stop'], start=False)
    return create_entity(start, end, entity=entity['entity'], value=entity['text'])


def messages_to_dataframe(messages: typing.Tuple[Message, ...], focus=src.typ.Focus.ALL) -> pd.DataFrame:
    """ Returns a DataFrame (table) from a list of Message objects which can be used for visualisation.

    Args:
        messages: Sentences. Sentence is annotated if (focus != 'intent')
        focus: Focus of the DataFrame.  For intent classification visualisation choose 'intent'
    """
    data = {'message': [], 'intent': [], 'training': []}
    for message in messages:
        intent_focus = focus == src.typ.Focus.INTENT
        data['message'].append(message.text if intent_focus else convert_message_to_annotated_str(message))
        data['intent'].append(message.data['intent'])
        data['training'].append(message.data['training'])
    return pd.DataFrame(data)


def generate_watson_intents(corpus: src.typ.Corpus, path: pathlib.Path):
    df = messages_to_dataframe(get_filtered_messages(corpus, train=True), src.typ.Focus.intent)
    df['intent'] = [s.replace(' ', '_') for s in df['intent']]
    df.drop('training', axis=1).to_csv(path, header=False, index=False)


def convert_index(text: str, token_index: int, start: bool) -> int:
    """ Convert token_index as used by NLU-Evaluation Corpora to character index. """
    span_generator = nltk.tokenize.WordPunctTokenizer().span_tokenize(text)
    spans = [span for span in span_generator]
    return spans[token_index][0 if start else 1]


def read_nlu_evaluation_corpora(js: dict, corpus: src.typ.Corpus) -> typing.Tuple[Message, ...]:
    """ Convert NLU Evaluation Corpora dictionary to the internal representation. """
    def convert_entities(sentence: dict) -> typing.List[dict]:
        return list(map(lambda e: convert_nlu_evaluation_entity(sentence['text'], e), sentence['entities']))

    def convert_sentence(sentence: dict) -> Message:
        return create_message(sentence['text'], sentence['intent'], convert_entities(sentence),
                              sentence['training'], corpus)

    return tuple(map(convert_sentence, js['sentences']))


def read_snips(js: dict) -> pd.DataFrame:
    """ Process some json file containing Snips corpus. """
    data = {'sentence': [], 'intent': [], 'training': []}

    queries_count = 0

    for domain in js['domains']:
        for intent in domain['intents']:
            for query in intent['queries']:
                queries_count += 1
                data['sentence'].append(query['text'])
                data['intent'].append(query['results_per_service']['Snips']['classified_intent'])
                data['training'].append(False)  # TODO: Fix this

    return pd.DataFrame(data)


@functools.lru_cache()
def get_messages(corpus: src.typ.Corpus) -> typing.Tuple[Message, ...]:
    """ Get all messages: Message from some file containing corpus and cache the messages. """
    if corpus == corpus.MOCK:
        return tuple(src.mock.get_mock_messages())

    file = src.utils.get_root() / 'datasets' / get_path(corpus)
    with open(str(file), 'rb') as f:
        js = json.load(f)

    parent_folder: pathlib.Path = file.parent
    if parent_folder.name == 'NLU-Evaluation-Corpora':
        return read_nlu_evaluation_corpora(js, corpus)
    elif parent_folder.name == 'snips':
        return read_snips(js)


def get_filtered_messages(corpus: src.typ.Corpus, train: bool) -> typing.Iterable[Message]:
    return filter(lambda m: train == m.data['training'], get_messages(corpus))


def get_intents(corpus: src.typ.Corpus) -> typing.Iterable[str]:
    """ Returns intent for each message in some corpus. To get unique intents one can simply cast it to a set. """
    return map(lambda m: m.data['intent'], get_messages(corpus))
