from nlu_analysers import analyser
from systems.rasa.rasa import Rasa
import json


def get_annotations(corpus, output, rasa: Rasa()):
    data = json.load(open(corpus))
    annotations = {'results': []}

    for s in data["sentences"]:
        if not s["training"]:
            annotations['results'].append(rasa.evaluate(s['text']))

    file = open(output, "wb")
    file.write(
        json.dumps(annotations, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False).encode(
            'utf-8'))
    file.close()


def analyse_annotations(annotations_file, corpus_file, output_file):
    analysis = {"intents": {}, "entities": {}}

    corpus = json.load(open(corpus_file))
    gold_standard = []
    for s in corpus["sentences"]:
        if not s["training"]:  # only use test data
            gold_standard.append(s)

    annotations = json.load(open(annotations_file))

    i = 0
    for a in annotations["results"]:
        if not a["text"] == gold_standard[i]["text"]:
            print(a["query"])
            print(gold_standard[i]["text"])
            print("WARNING! Texts not equal")

        # intent
        a_intent = a['intent']['name']
        o_intent = gold_standard[i]["intent"]

        analyser.check_key(analysis["intents"], a_intent)
        analyser.check_key(analysis["intents"], o_intent)

        if a_intent == o_intent:
            # correct
            analysis["intents"][a_intent]["truePos"] += 1
        else:
            # incorrect
            analysis["intents"][a_intent]["falsePos"] += 1
            analysis["intents"][o_intent]["falseNeg"] += 1

        # entities
        a_entities = a["entities"]
        o_entities = gold_standard[i]["entities"]

        for x in a_entities:
            analyser.check_key(analysis["entities"], x["type"])

            if len(o_entities) < 1:  # false pos
                analysis["entities"][x["type"]]["falsePos"] += 1
            else:
                true_pos = False

                for y in o_entities:
                    if Luisanalyser.detokenizer(x["entity"]) == y["text"].lower():
                        if x["type"] == y["entity"]:  # truePos
                            true_pos = True
                            o_entities.remove(y)
                            break
                        else:  # falsePos + falseNeg
                            analysis["entities"][x["type"]]["falsePos"] += 1
                            analysis["entities"][y["entity"]]["falseNeg"] += 1
                            o_entities.remove(y)
                            break
                if true_pos:
                    analysis["entities"][x["type"]]["truePos"] += 1
                else:
                    analysis["entities"][x["type"]]["falsePos"] += 1

        for y in o_entities:
            analyser.check_key(analysis["entities"], y["entity"])
            analysis["entities"][y["entity"]]["falseNeg"] += 1

        i += 1

        analyser.write_json(output_file,
                            json.dumps(analysis, sort_keys=False, indent=4, separators=(',', ': '),
                                       ensure_ascii=False).encode('utf-8'))
