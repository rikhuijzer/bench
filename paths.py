from pathlib import Path


class Paths:
    root = Path(__file__).parent

    def __init__(self, corpus, system):
        self.corpus = corpus
        self.system = system

    def folder_generated(self):
        return self.root / 'generated' / self.system

    def folder_corpora(self):
        return self.root / 'datasets' / 'NLU-Evaluation-Corpora'

    def file_corpus(self):
        return self.corpora() / str(self.corpus + '.json')
