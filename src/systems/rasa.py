import rasa_nlu.training_data
import requests
import src.system
import src.dataset
import src.typ
import json
from rasa_nlu.training_data import Message


# example output: https://rasa.com/docs/nlu/0.13.7/choosing_pipeline/#choosing-pipeline

def clear_data_field(message: Message) -> Message:
    """Clear my added corpus data field since it is not used and will cause problems for as_json()."""
    message.data['corpus'] = ''
    return message


def train(system_corpus: src.typ.SystemCorpus) -> src.typ.System:
    training_examples = src.dataset.get_filtered_messages(system_corpus.corpus, train=True)
    training_examples = list(map(lambda message: clear_data_field(message), training_examples))
    training_data = rasa_nlu.training_data.TrainingData(training_examples=training_examples).as_json()
    url = 'http://localhost:{}/train?project=my_project'

    # https://github.com/requests/requests/issues/1822
    training_data = training_data.encode('utf-8')

    r = requests.post(url.format(src.system.get_port(system_corpus.system.name)),
                      data=training_data, headers=src.system.get_header(src.typ.Header.JSON)).json()
    if 'error' in r:
        # print(str(training_data)[344:380].encode("utf-8"))
        raise RuntimeError('Training {} failed on {}, Response: \n {}.'.format(system_corpus.system.name,
                                                                               system_corpus.corpus, r))
    return src.typ.System(system_corpus.system.name, system_corpus.corpus, system_corpus.system.timestamp, ())


def get_response(query: src.typ.Query) -> src.typ.Response:
    data = {'q': query.text, 'project': 'my_project'}
    url = 'http://localhost:{}/parse'
    r = requests.post(url.format(src.system.get_port(query.system.name)),
                      data=json.dumps(data), headers=src.system.get_header(src.typ.Header.JSON))
    if r.status_code != 200:
        raise RuntimeError('Could not get intent for text: {}'.format(query.text))

    js = r.json()
    confidence = round(float(js['intent']['confidence']), 3)
    entities = js['entities'] if 'entities' in js else []
    return src.typ.Response(js['intent']['name'], confidence, entities)
