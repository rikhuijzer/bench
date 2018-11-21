from src.converters import get_split


def test_get_split():
    X = list(range(10))
    y = ['A'] * 7 + ['C'] * 3
    train, dev, test = get_split(X, y)
    assert 6 == len(train.y)
    assert 2 == len(dev.y)
    assert 2 == len(test.y)
