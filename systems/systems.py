import json
from abc import ABC, abstractmethod

import requests


class System(ABC):
    HEADER_JSON = {'content-type': 'application/json'}
    HEADER_YML = {'content-type': 'application/x-yml'}

    port: int

    def __init__(self, port: int):
        self.port = port

    @abstractmethod
    def get_intent(self, sentence: str) -> str:
        pass


class Rasa(System):
    trained = False

    def __init__(self, port: int):
        super().__init__(port)

    def train(self):
        print('Training Rasa...')
        data = {'q': 'b'}
        url = 'http://localhost:{}/train?project=my_project'
        r = requests.post(url.format(self.port), data=json.dumps(data), headers=self.HEADER_JSON).json()
        print('Response: ' + str(r))
        if 'error' in r:
            raise RuntimeError('Training failed.')
        self.trained = True

    def get_intent(self, sentence: str):
        if not self.trained:
            self.train()

        data = {'q': sentence}
        url = 'http://localhost:{}/parse'
        r = requests.post(url.format(self.port), data=json.dumps(data), headers=self.HEADER_JSON)
        return r.json()['intent']['name']


class DeepPavlov(System):
    def __init__(self, port: int):
        super().__init__(port)

    def get_intent(self, sentence: str):
        data = {'context': [sentence]}
        r = requests.post('http://localhost:{}/intent'.format(self.port), data=json.dumps(data),
                          headers=self.HEADER_JSON)
        return r.json()[0][0][0]


class Watson(System):
    # possibly interesting: https://github.com/joe4k/wdcutils/
    def __init__(self, url):
        super().__init__(url)

    def get_intent(self, sentence: str):
        print('unimplemented')
