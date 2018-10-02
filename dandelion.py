import pandas
import tensorflow as tf
import os
import requests


def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            # (sent_1, sent_2, similarity_score)
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dataset = tf.keras.utils.get_file(
      fname="Stsbenchmark.tar.gz",
      origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
      extract=True)

    sts_dev = load_sts_dataset(
      os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(
      os.path.join(
          os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


sts_dev, sts_test = download_and_load_sts_data()

token = '407d0501c07244f098026aed38cb1fcc'
# text1 = 'Cameron wins the scar'
# text2 = 'All nominees for the Academy Awards'
text1 = 'mijn telefoon is stuk'
text2 = 'waarom doet mijn mobiel het niet?'

url = 'https://api.dandelion.eu/datatxt/sim/v1/?text1=' + text1 +\
          '&text2=' + text2 + '&token=' + token

contents = requests.get(url)
print(contents.json())