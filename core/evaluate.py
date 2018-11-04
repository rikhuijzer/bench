from systems.systems import *
import core.typ
import sklearn.metrics
import core.training_data
import typing


def classify_intents(system: core.typ.System, corpus: core.typ.Corpus) -> core.typ.Classifications:
    """ Run all test sentences from some corpus through system and return results. """
    messages = core.training_data.get_filtered_messages(corpus, core.typ.TrainTest.test)
    df = core.training_data.messages_to_dataframe(messages)

    classifications = []
    confidences = []
    for _, row in df.iterrows():
        system, response = get_intent(system, core.typ.TestSentence(row['message'], corpus))
        classifications.append(response.intent)
        confidences.append(response.confidence)  # TODO: Add this information to DataFrame?

    df['classification'] = classifications
    return core.typ.Classifications(system, df)


def get_f1_score(system: core.typ.System, corpus: core.typ.Corpus, average='micro') -> core.typ.F1Scores:
    """ Get f1 score for some system and corpus. Based on scikit-learn f1 score calculation. """
    system, df = classify_intents(system, corpus)
    score = round(sklearn.metrics.f1_score(df['intent'], df['classification'], average=average), 3)
    return core.typ.F1Scores(system, (score,))


def get_f1_score_runs(system: core.typ.System, corpus: core.typ.Corpus,
                      n_runs: int, average='micro') -> typing.Tuple[float, ...]:
    """ Get f1 score multiple times and re-train system each time. """
    out = []
    for i in range(0, n_runs):
        system, scores = get_f1_score(
            core.typ.System(system.name, system.knowledge,
                   system.data + ('retrain', ) if i > 0 else system.data), corpus, average)
        out.append(scores)
    return tuple(out)

# s1, 0 = fn(s0)
# s2, 1 = fn(s1)
# s3, 2 = fn(s2)
# the problem can be solved by having a function which generates states and a function which gives an output for
# a state. Currently the function does two things, causing it not to be modular
# if function is split into two, the states can be generated (accumulate probably) and then using map.
# however, it is not useful here since system state and output is coupled by stateful servers :\
