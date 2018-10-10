import json
from pathlib import Path


class Converter(object):
    @staticmethod
    def tokenizer(s):
        return s.replace(".", " . ").replace(",", " , ").replace("'", " ' ").replace("?", " ? ").replace("!",
                                                                                                         " ! ").replace(
            "&", " & ").replace(":", " : ").replace("-", " - ").replace("/", " / ").replace("(", " ( ").replace(")",
                                                                                                                " ) ").replace(
            "  ", " ").split(" ")

    @staticmethod
    def detokenizer(s):
        return s.replace(" . ", ".").replace(" , ", ",").replace(" ' ", "'").replace(" ? ", "?").replace(" ! ",
                                                                                                         "!").replace(
            " & ", "&").replace(" : ", ":").replace(" - ", "-").replace(" / ", "/").replace(" ( ", "(").replace(" ) ",
                                                                                                                ")")

    @staticmethod
    def write_json(file, content):
        print(content)
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        with open(file, 'w') as f:
            json.dump(content, f)

    @staticmethod
    def array_to_json(array):
        j = []
        for e in array:
            j.append({"name": e})
        return j

    def __init__(self):
        self.intents = set()
        self.entities = set()
        self.utterances = []

        self.name = ""
        self.desc = ""
        self.lang = ""

    def __add_intent(self, intent):
        raise NotImplementedError("Please implement this method")

    def __add_entity(self, entity):
        raise NotImplementedError("Please implement this method")

    def __add_utterance(self, sentence):
        raise NotImplementedError("Please implement this method")

    def import_corpus(self, file):
        raise NotImplementedError("Please implement this method")

    def export(self):
        raise NotImplementedError("Please implement this method")
