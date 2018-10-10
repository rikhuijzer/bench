# load all test data from corpus
# check output from system
# validate whether output from system is correct
from corpus import Corpus
from systems.rasa.rasa import Rasa


def test(corpus: Corpus, rasa: Rasa):
    df = corpus.get_test()

    output = []
    for _, row in df.iterrows():
        intent = rasa.get_intent(row['sentence'])['name']
        output.append(intent)

    df['guess'] = output
    print(df)
