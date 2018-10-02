import requests
from system import System


class Dandelion(System):
    token = '407d0501c07244f098026aed38cb1fcc'

    def __init__(self):
        System.__init__(self)

    def get_sts(self, text1, text2):
        url = 'https://api.dandelion.eu/datatxt/sim/v1/?text1=' + text1 +\
            '&text2=' + text2 + '&token=' + self.token

        contents = requests.get(url)
        print(contents.json())
