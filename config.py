import configparser
from pathlib import Path


def get_value(section, key):
    config_file = Path(__file__).parent / 'config.ini'
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config[section][key]
