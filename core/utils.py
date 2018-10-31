import collections
import math
from time import time
from pathlib import Path


def get_root() -> Path:
    """ All path references should refer to this function to decouple from caller source file location. """
    return Path(__file__).parent.parent


def is_hashable(obj) -> bool:
    return isinstance(obj, collections.Hashable)


def square_ceil(x: int) -> int:
    return 1 if x == 0 else 2**math.ceil(math.log2(x))


def timer(fn):
    """ Using closure to measure running time for some function fn. """
    def inner(*args, **kwargs):
        t = time()
        fn(*args, **kwargs)
        print("took {time}".format(time=time()-t))
