import collections
import logging
import math
import pathlib
import time
import types
import typing
import datetime
import src.typ


def get_root() -> pathlib.Path:
    """ All path references should refer to this function to decouple from caller source file location. """
    return pathlib.Path(__file__).parent.parent


def is_hashable(obj) -> bool:
    return isinstance(obj, collections.Hashable)


def square_ceil(x: int) -> int:
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


def timer(fn: types.FunctionType) -> types.FunctionType:
    """ Using closure to measure running time for some function fn. """

    def inner(*args, **kwargs):
        t = time.time()
        fn(*args, **kwargs)
        print("took {time}".format(time=time.time() - t))

    return inner


T = typing.TypeVar('T')


def get_substring_match(dictionary: dict, string: str) -> T:
    """ Returns dictionary value for first key which is a substring of string.  """
    matches = list(filter(lambda key: key in string.lower(), dictionary))
    n = len(matches)

    if n == 0:
        raise ValueError('Tried to find one key which is a substring of {} but found 0.'.format(string))
    if n > 1:
        logging.warning('Tried to find one key which is a substring of {} but found {}: {}.'
                        .format(string, n, matches))

    return dictionary[matches[0]]


timestamp_format = '%Y-%m-%d %H:%M:%S'


def get_timestamp() -> str:
    """Returns timestamp as string. Miliseconds are removed by ignoring last few characters of string."""
    return str(datetime.datetime.now())[:-7]


def convert_str_timestamp(timestamp: str) -> datetime.datetime:
    return datetime.datetime.strptime(timestamp, timestamp_format)
