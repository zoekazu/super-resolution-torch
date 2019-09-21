
from logger import get_logger


def run():
    logger = get_logger(__file__)
    logger.info('information')


if __name__ == '__main__':
    run()
