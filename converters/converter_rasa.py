import stringcase

from converters.annotated_sentence import AnnotatedSentence
from converters.converter import *


class ConverterRasa(Converter):
    def __init__(self, paths):
        super(ConverterRasa, self).__init__()
        self.paths = paths
        self.training_file = paths.folder_generated() / 'training.md'

    def __add_intent(self, intent):
        self.intents.add(intent)

    def __add_entity(self, entity):
        self.entities.add(entity)

    def __add_utterance(self, sentence):
        entities = []
        for e in sentence.entities:
            entities.append({"entity": e["entity"], "startPos": e["start"], "endPos": e["stop"]})
        self.utterances.append({"text": sentence.text, "intent": sentence.intent, "entities": entities})

    def import_corpus(self):
        data = json.load(open(self.paths.file_corpus()))

        for s in data["sentences"]:
            if s["training"]:
                self.__add_intent(s["intent"])
                for e in s["entities"]:
                    self.__add_entity(e["entity"])
                self.__add_utterance(AnnotatedSentence(s["text"], s["intent"], s["entities"]))

    def export(self):
        with open(self.training_file, 'w') as f:
            for intent in self.intents:
                snake_case = stringcase.snakecase(intent)
                snake_case = snake_case.replace('__', '_')
                f.write('## intent:' + snake_case + '\n')
                for utterance in self.utterances:
                    if utterance['intent'] == intent:
                        f.write('- ' + utterance['text'] + '\n')
                f.write('\n')
