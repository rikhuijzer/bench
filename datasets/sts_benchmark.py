from data import Data
import pandas as pd
from pathlib import Path
import tensorflow as tf


class STSBenchmark(Data):
    def __init__(self):
        Data.__init__(self)

    def get_lines(self):
        file = Path(__file__).parent / 'sts_benchmark_files' / 'sts-test.csv'
        sts_test = self.load_sts_dataset(file)
        print(sts_test)
        return []

    @staticmethod
    def load_sts_dataset(filename):
        filename = str(filename)
        # Loads a subset of the STS dataset into a DataFrame. In particular both
        # sentences and their human rated similarity score.
        sent_pairs = []
        with tf.gfile.GFile(filename, "r") as f:
            for line in f:
                ts = line.strip().split("\t")
                # (sent_1, sent_2, similarity_score)
                sent_pairs.append((ts[5], ts[6], float(ts[4])))
        return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])
