from systems.dandelion import Dandelion
from datasets.sts_benchmark import STSBenchmark
# TODO: Loop over each sentence and then each system, this way the API calls do not need a sleep statement
# TODO: Integrate database to help with displaying accuracies in table including test date. Re-test monthly?


def test_dandelion():
    s = Dandelion()

    text1 = 'Cameron wins the scar'
    text2 = 'All nominees for the Academy Awards'

    s.get_sts(text1, text2)


def test_sts_benchmark():
    b = STSBenchmark()
    lines = b.get_lines()
    print(lines)


if __name__ == '__main__':
    2
