from abc import ABC, abstractmethod


class Data(ABC):
    def __init__(self):
        super(Data, self).__init__()

    @abstractmethod
    def get_lines(self):
        pass
