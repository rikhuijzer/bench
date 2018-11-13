import rasa_nlu.training_data
import requests
import src.system
import src.dataset
import src.typ as tp
import json
from rasa_nlu.training_data import Message
import os
from typing import Tuple
from src.dataset import get_intents, get_filtered_messages

# see https://github.com/GoogleCloudPlatform/python-docs-samples/tree/master/dialogflow/cloud-client
# since documentation seems missing for V2


def create_intent(project_id, display_name, training_phrases_parts,
                  message_texts):
    """Create an intent of the given intent type."""

    import dialogflow_v2 as dialogflow
    intents_client = dialogflow.IntentsClient()

    parent = intents_client.project_agent_path(project_id)
    training_phrases = []
    for training_phrases_part in training_phrases_parts:
        part = dialogflow.types.Intent.TrainingPhrase.Part(
            text=training_phrases_part)
        # Here we create a new training phrase for each provided part.
        training_phrase = dialogflow.types.Intent.TrainingPhrase(parts=[part])
        training_phrases.append(training_phrase)

    text = dialogflow.types.Intent.Message.Text(text=message_texts)
    message = dialogflow.types.Intent.Message(text=text)

    intent = dialogflow.types.Intent(
        display_name=display_name,
        training_phrases=training_phrases,
        messages=[message])

    response = intents_client.create_intent(parent, intent)

    print('Intent created: {}'.format(response))


def train(system_corpus: src.typ.SystemCorpus) -> src.typ.System:
    messages = get_filtered_messages(system_corpus.corpus, train=True)
    intents = set(get_intents(system_corpus.corpus))

    for intent in intents:
        filtered_messages = filter(lambda m: m.data['intent'] == intent, messages)
        texts = tuple(map(lambda m: m.text, filtered_messages))
        create_intent('bench-9bcea', intent, texts, [intent])
    return src.typ.System(system_corpus.system.name, system_corpus.corpus, system_corpus.system.timestamp, ())


def get_response(query: src.typ.Query) -> src.typ.Response:
    data = {'q': query.text, 'project': 'my_project'}
    r = requests.post(url.format(src.system.get_port(query.system.name)),
                      data=json.dumps(data), headers=src.system.get_header(src.typ.Header.JSON))
    if r.status_code != 200:
        raise RuntimeError('Could not get intent for text: {}'.format(query.text))
    return src.typ.Response(r.json()['intent']['name'], '-1.0', [])