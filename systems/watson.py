import core.typ
import os


# Possibly interesting: https://github.com/joe4k/wdcutils/


def get_response(query: core.typ.Query) -> core.typ.Response:
    from watson_developer_cloud import AssistantV1
    import time
    time.sleep(1)
    default_url = 'https://gateway.watsonplatform.net/assistant/api'  # might differ based on workspace
    assistant = AssistantV1(version='2018-09-20', username=os.environ['WATSON_USERNAME'],
                            password=os.environ['WATSON_PASSWORD'], url=default_url)
    response = assistant.message(workspace_id='c6548076-8034-4f28-a155-ab546b0058d5',
                                 input={'text': query.text},
                                 alternate_intents=False).get_result()
    print(response['intents'])
    if response['intents']:
        classification = response['intents'][0]['intent'].replace('_', ' ')
    else:
        classification = ''
    return core.typ.Classification(classification, '-1.0', [])
