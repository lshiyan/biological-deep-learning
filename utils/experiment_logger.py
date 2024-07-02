import logging


DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = logging.Formatter('%(asctime)s || %(levelname)-10s || %(lineno)-4d || %(funcName)-15s || %(message)s')
DEFAULT_FILE = 'error.log'
DEFAULT_NAME = 'Error Log'


def configure_logger(name: str, file: str, level: int = DEFAULT_LEVEL, format: str = DEFAULT_FORMAT) -> logging.Logger:
    """
    FUNCTION
    Creates a logger to log information
    @param
        name: name of logger
        file: path to file
        level: level of the log
        format: format of the log
    @return
        logger: a logger
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(level)
    handler: logging.Handler = logging.FileHandler(file)
    handler.setLevel(level)
    handler.setFormatter(format)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def close_logger(logger: logging.Logger) -> None:
    """
    FUNCTION
    Closes and removes all handlers and loggers
    @param
        logger: logger to be closed
    @return
        None
    """
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)