import rasa_nlu.training_data
import requests
import src.system
import src.dataset
import src.typ as tp
import json
from rasa_nlu.training_data import Message
import dialogflow
import os
from typing import Tuple


url = 'https://api.dialogflow.com/'


def get_template() -> str:
    """Returns a default template."""
    return '''
    {
      "contexts": [],
      "events": [],
      "fallbackIntent": false,
      "name": "<intent_name>",
      "priority": 500000,
      "responses": [
        {
          "resetContexts": false,
          "affectedContexts": [],
          "parameters": [],
          "messages": [
            {
              "type": 0,
              "lang": "en",
              "speech": []
            }
          ],
          "defaultResponsePlatforms": {},
          "speech": []
        }
      ],
      "userSays": [
          {
            "id": "8f7e3a72-f0bc-4e95-b737-c1c4434de601",
            "data": [
              {
                "text": "intent1_utterance2",
                "userDefined": false
              }
            ],
            "isTemplate": false,
            "count": 0,
            "updated": 1542036539
          },
          {
            "id": "97681070-3320-4190-a52e-771d97657ba9",
            "data": [
              {
                "text": "intent1_utterance1",
                "userDefined": false
              }
            ],
            "isTemplate": false,
            "count": 0,
            "updated": 1542036539
          }
        ],
      "webhookForSlotFilling": false,
      "webhookUsed": false
    }
    '''


def get_user_say(utterance: str) -> dict:
    return {
        'data': [{'text': utterance, 'userDefined': False}],
        'isTemplate': False,
        'count': 0,
    }


def fill_json(intent: str, utterances: Tuple[str, ...]) -> dict:
    js = json.loads(get_template())
    js['name'] = intent
    js['userSays'] = []
    for utterance in utterances:
        js['userSays'].append(get_user_say(utterance))
    return js


def get_token() -> str:
    return os.environ['DIALOGFLOW_DEV_TOKEN']


def get_header() -> dict:
    return {
        'Authorization': 'Bearer {}'.format(get_token()),
        'Content-Type': 'application_json'
    }


def train(system_corpus: src.typ.SystemCorpus) -> src.typ.System:
    intent_url = url + 'v1/intents?v=20150910'  # this is the version used in the api docs
    r = requests.post(intent_url, data=json.dumps(fill_json('intent1', ('utter1', 'utter2'))), headers=get_header())
    print(r)
    return src.typ.System(system_corpus.system.name, system_corpus.corpus, system_corpus.system.timestamp, ())


def get_response(query: src.typ.Query) -> src.typ.Response:
    data = {'q': query.text, 'project': 'my_project'}
    r = requests.post(url.format(src.system.get_port(query.system.name)),
                      data=json.dumps(data), headers=src.system.get_header(src.typ.Header.JSON))
    if r.status_code != 200:
        raise RuntimeError('Could not get intent for text: {}'.format(query.text))
    return src.typ.Response(r.json()['intent']['name'], '-1.0', [])