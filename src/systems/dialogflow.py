import requests
import src.system
import src.dataset
import src.typ as tp
import json
from src.dataset import get_intents, get_filtered_messages
import json

import requests

import src.dataset
import src.system
import src.typ as tp
from src.dataset import get_intents, get_filtered_messages
from google.api_core import exceptions


# see https://github.com/GoogleCloudPlatform/python-docs-samples/tree/master/dialogflow/cloud-client
# since documentation seems missing for V2

project_id = 'bench-9bcea'


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
    messages = tuple(get_filtered_messages(system_corpus.corpus, train=True))
    intents = set(get_intents(system_corpus.corpus))

    for intent in intents:
        filtered_messages = filter(lambda m: m.data['intent'] == intent, messages)
        texts = tuple(map(lambda m: m.text, filtered_messages))

        try:  # can also get list of intents first
            create_intent(project_id, intent, texts, [intent])
        except exceptions.FailedPrecondition:
            print('Failed to create intent {}. Skipping'.format(intent))
    return src.typ.System(system_corpus.system.name, system_corpus.corpus, system_corpus.system.timestamp, ())


def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.
    Using the same `session_id` between requests allows continuation
    of the conversaion."""
    import dialogflow_v2 as dialogflow
    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session))

    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)

        print('=' * 20)
        print('Query text: {}'.format(response.query_result.query_text))
        print('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        print('Fulfillment text: {}\n'.format(
            response.query_result.fulfillment_text))
        return response.query_result.intent.display_name


def get_response(query: src.typ.Query) -> src.typ.Response:
    """Returns the result of detect intent with texts as inputs.
    Using the same `session_id` between requests allows continuation
    of the conversaion."""
    import dialogflow_v2 as dialogflow
    session_client = dialogflow.SessionsClient()

    session_id = None
    session = session_client.session_path(project_id, session_id)

    text_input = dialogflow.types.TextInput(text=query.text, language_code='en')
    query_input = dialogflow.types.QueryInput(text=text_input)
    response = session_client.detect_intent(session=session, query_input=query_input)

    intent = response.query_result.intent.display_name
    confidence = response.query_result.intent_detection_confidence
    return src.typ.Response(intent, confidence, [])
