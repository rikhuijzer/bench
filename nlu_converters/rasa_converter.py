import stringcase

from nlu_converters.annotated_sentence import AnnotatedSentence
from nlu_converters.converter import *


class RasaConverter(Converter):
    LUIS_SCHEMA_VERSION = "1.3.0"

    def __init__(self):
        super(RasaConverter, self).__init__()
        self.bing_entities = set()
        self.training_file = Path(__file__).parent.parent / 'generated' / 'rasa' / 'training.md'

    def __add_intent(self, intent):
        self.intents.add(intent)

    def __add_entity(self, entity):
        self.entities.add(entity)

    def __add_bing_entity(self, entity):
        self.bing_entities.add(entity)

    def __add_utterance(self, sentence):
        entities = []
        for e in sentence.entities:
            entities.append({"entity": e["entity"], "startPos": e["start"], "endPos": e["stop"]})
        self.utterances.append({"text": sentence.text, "intent": sentence.intent, "entities": entities})

    def import_corpus(self, corpus):
        data = json.load(open(corpus))

        # training data
        for s in data["sentences"]:
            if s["training"]:  # only import training data
                # intents
                self.__add_intent(s["intent"])
                # entities
                for e in s["entities"]:
                    self.__add_entity(e["entity"])
                # utterances
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
                print(self.training_file)
