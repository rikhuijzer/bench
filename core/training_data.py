import json
import functools
import typing
import pandas as pd
import nltk.tokenize
import rasa_nlu.training_data
import rasa_nlu.training_data.formats.markdown
import rasa_nlu.utils
import core.typ
import core.utils
import pathlib

pd.set_option('max_colwidth', 180)


def create_entity(start: int, end: int, entity: str, value: str) -> dict:
    return {'start': start, 'end': end, 'entity': entity, 'value': value}


def create_message(text: str, intent: str, entities: [], training: bool) -> rasa_nlu.training_data.Message:
    """ Helper function to create a message: Message used by Rasa including whether train or test sentence. """
    message = rasa_nlu.training_data.Message.build(text, intent, entities)
    message.data['training'] = training
    return message


def convert_message_to_annotated_str(message: rasa_nlu.training_data.Message) -> str:
    """ Convert Message object to string having annotated entities. """
    training_examples: typing.List[rasa_nlu.training_data.Message] = [message]
    training_data = rasa_nlu.training_data.TrainingData(training_examples=training_examples)
    generated = rasa_nlu.training_data.formats.markdown.MarkdownWriter()._generate_training_examples_md(training_data)
    generated = generated[generated.find('\n') + 3:-1]  # removes header
    return generated


def convert_nlu_evaluation_entity(text: str, entity: dict) -> dict:
    """ Convert a NLU Evaluation Corpora sentence to Entity object. See test for examples. """
    start = convert_index(text, entity['start'], core.typ.StartEnd.start)
    end = convert_index(text, entity['stop'], core.typ.StartEnd.end)
    return create_entity(start, end, entity=entity['entity'], value=entity['text'])


def messages_to_dataframe(messages: typing.Tuple[rasa_nlu.training_data.Message, ...], focus=core.typ.Focus.all) -> pd.DataFrame:
    """ Returns a DataFrame (table) from a list of Message objects which can be used for visualisation.

    Args:
        messages: Sentences. Sentence is annotated if (focus != 'intent')
        focus: Focus of the DataFrame.  For intent classification visualisation choose 'intent'
    """
    data = {'message': [], 'intent': [], 'training': []}
    for message in messages:
        data['message'].append(message.text if focus.value == 'intent' else convert_message_to_annotated_str(message))
        data['intent'].append(message.data['intent'])
        data['training'].append(message.data['training'])
    return pd.DataFrame(data)


def generate_watson_intents(corpus: core.typ.Corpus, path: pathlib.Path):
    df = messages_to_dataframe(get_filtered_messages(corpus, core.typ.TrainTest.train), core.typ.Focus.intent)
    df['intent'] = [s.replace(' ', '_') for s in df['intent']]
    df.drop('training', axis=1).to_csv(path, header=False, index=False)


def convert_index(text: str, token_index: int, start_end: core.typ.StartEnd) -> int:
    """ Convert token_index as used by NLU-Evaluation Corpora to character index. """
    span_generator = nltk.tokenize.WordPunctTokenizer().span_tokenize(text)
    spans = [span for span in span_generator]
    return spans[token_index][start_end.value]


def read_nlu_evaluation_corpora(js: dict) -> core.typ.Messages:
    """ Convert NLU Evaluation Corpora dictionary to the internal representation. """
    def convert_entities(sentence: dict) -> typing.List[dict]:
        return list(map(lambda e: convert_nlu_evaluation_entity(sentence['text'], e), sentence['entities']))

    def convert_sentence(sentence: dict) -> rasa_nlu.training_data.Message:
        return create_message(sentence['text'], sentence['intent'], convert_entities(sentence), sentence['training'])

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


def filter_train_test(messages: typing.Tuple, train_test: core.typ.TrainTest) -> core.typ.Messages:
    """ Get train or test split for some corpus. """
    return tuple([message for message in messages if message.data['training'] == train_test.value])


@functools.lru_cache(maxsize=core.utils.square_ceil(len(core.typ.Corpus)))
def get_messages(corpus: core.typ.Corpus) -> typing.Tuple:
    """ Get all messages: Message from some file containing corpus and cache the messages. """
    if corpus == corpus.Mock:
        return tuple(map(lambda x: create_message(str(x), 'A' if 0 <= x < 10 else 'B', [], False), range(0, 20)))

    file = pathlib.Path(__file__).parent.parent / 'datasets' / corpus.value
    with open(str(file), 'rb') as f:
        js = json.load(f)

    parent_folder: pathlib.Path = file.parent
    if parent_folder.name == 'NLU-Evaluation-Corpora':
        return read_nlu_evaluation_corpora(js)
    elif parent_folder.name == 'snips':
        return read_snips(js)


def get_filtered_messages(corpus: core.typ.Corpus, train_test: core.typ.TrainTest) -> core.typ.Messages:
    return filter_train_test(get_messages(corpus), train_test)


def get_intents(corpus: core.typ.Corpus, train_test: core.typ.TrainTest) -> typing.Set[str]:
    return set(map(lambda m: m.data['intent'], get_filtered_messages(corpus, train_test)))
