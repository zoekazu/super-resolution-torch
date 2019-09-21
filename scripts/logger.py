"""logger configlation


<explanation>
"""

import logging

logger = logging.getLogger('log')

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(filename='log.txt', mode='w')
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

handler_formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(filename)s: %(message)s')
stream_handler.setFormatter(handler_formatter)

logger.addHandler(stream_handler)


def get_logger(logger_name: str):
    return logger.getChild(logger_name)
