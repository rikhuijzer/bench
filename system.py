from abc import ABC, abstractmethod


class System(ABC):
    def __init__(self):
        super(System, self).__init__()

    @abstractmethod
    def get_sts(self, text1, text2):
        pass
