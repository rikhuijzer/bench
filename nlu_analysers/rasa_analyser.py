from nlu_analysers.luis_analyser import *
from systems.rasa.rasa import Rasa


class RasaAnalyser:  # not extending luis anymore since inheritance is just plain annoying
    def __init__(self):
        self.luis_analyser = LuisAnalyser()

    def get_annotations(self, corpus, output, rasa: Rasa()):
        data = json.load(open(corpus))
        annotations = {'results': []}

        for s in data["sentences"]:
            if not s["training"]:  # only use test data
                # encoded_text = urllib.parse.quote(s['text'])
                # annotations['results'].append(requests.get(self.url % encoded_text, data={}, headers={}).json())
                annotations['results'].append(rasa.evaluate(s['text']))

        file = open(output, "wb")
        file.write(
            json.dumps(annotations, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False).encode(
                'utf-8'))
        file.close()

    def analyse_annotations(self, annotations_file, corpus_file, output_file):
        self.luis_analyser.analyse_annotations(self, annotations_file, corpus_file, output_file)
