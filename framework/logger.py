import logging

debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name) - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(message)s'))


def get_logger(logfile_path=None) -> logging.Logger:
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    if logfile_path:
        file_handler = logging.FileHandler(logfile_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(debug_formatter)
        logger.addHandler(file_handler)

    logger.addHandler(stream_handler)

    return logger


log = get_logger()
