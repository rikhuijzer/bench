import unittest
import pandas as pd
import evaluate


class TestEvaluate(unittest.TestCase):

    def test_count_true(self):
        df = pd.DataFrame(data={'col1': [True, False], 'col2': [False, False], 'col3': [True, True]})
        self.assertEqual(evaluate.count_true(df, 'col1'), 1)
        self.assertEqual(evaluate.count_true(df, 'col2'), 0)
