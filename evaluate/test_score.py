import unittest
from evaluate.score import F1Score


class TestF1Score(unittest.TestCase):

    def test(self):
        score = F1Score()
        score.add_classification('foo', 'foo')


if __name__ == '__main__':
    unittest.main()
