import logging


#########################################################################
# CONSTANTS
#########################################################################
DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = logging.Formatter('%(asctime)s || %(levelname)-10s || %(lineno)-4d || %(funcName)-15s || %(message)s')
DEFAULT_FILE = 'error.log'
DEFAULT_NAME = "Error Log"


#########################################################################
# Logging functions
#########################################################################
"""
Method to create a logger to log information
@param
    name (str) = name of logger
    file (str) - path to file
    level (logging.Level) = level of the log
    format (logging.Formatter) = format of the log
@return
    logger (logging.Logger) = a logger
"""
def configure_logger(name, file, level=DEFAULT_LEVEL, format=DEFAULT_FORMAT):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(file)
    handler.setLevel(level)
    handler.setFormatter(format)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def get_print_log(name, path, level=DEFAULT_LEVEL, format=DEFAULT_FORMAT):
    return configure_logger(name, path+"/prints.log", level, format)

def get_parameter_log(name, path, level=DEFAULT_LEVEL, format=DEFAULT_FORMAT):
    return configure_logger(name, path+"/parameters.log", level, format)

def get_test_acc_log(name, path, level=DEFAULT_LEVEL, format=DEFAULT_FORMAT):
    return configure_logger(name, path+"/testing_accuracy.log", level, format)

def get_train_acc_log(name, path, level=DEFAULT_LEVEL, format=DEFAULT_FORMAT):
    return configure_logger(name, path+"/training_accuracy.log", level, format)

def get_debug_log(name, path, level=logging.DEBUG, format=DEFAULT_FORMAT):
    return configure_logger(name, path+"/debug.log", level, format)

def get_experiment_log(name, path, level=DEFAULT_LEVEL, format=DEFAULT_FORMAT):
    return configure_logger(name, path+"/experiment_process.log", level, format)

