from abc import ABC, abstractmethod
import requests
import json


class System(ABC):
    url: str

    def __init__(self, url):
        self.url = url

    @abstractmethod
    def get_intent(self, sentence: str) -> str:
        pass


class DeepPavlov(System):
    def __init__(self, url):
        super().__init__(url)

    def get_intent(self, sentence: str):
        data = {'context': [sentence]}
        headers = {'content-type': 'application/json'}
        response = requests.post(self.url, data=json.dumps(data), headers=headers)
        print(response)
        r = requests.get(self.url)
        print(r.json())
        return 2


class Watson(System):
    # possibly interesting: https://github.com/joe4k/wdcutils/
    def __init__(self, url):
        super().__init__(url)

    def get_intent(self, sentence: str):
        print('unimplemented')
