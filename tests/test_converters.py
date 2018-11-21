from src.converters import get_split


def test_get_split():
    def about_equal(a: int, b: int) -> bool:
        return abs(a - b) < 5

    X = list(range(100))
    y = ['A'] * 70 + ['B'] * 30
    train, dev, test = get_split(X, y)
    assert about_equal(60, len(train.y))
    assert about_equal(20, len(dev.y))
    assert about_equal(20, len(test.y))
