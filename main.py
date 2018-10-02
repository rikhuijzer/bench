from systems.Dandelion import Dandelion
from system import System

# TODO: Loop over each sentence and then each system, this way the API calls do not need a sleep statement


if __name__ == '__main__':
    s: System
    s = Dandelion()

    text1 = 'Cameron wins the scar'
    text2 = 'All nominees for the Academy Awards'

    s.get_sts(text1, text2)
