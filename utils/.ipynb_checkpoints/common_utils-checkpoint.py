import logging


import json

def create_logger(log_file=None, log_level=logging.INFO):
    """
    Creates and configures a logger object.

    Args:
        log_file (str, optional): The path to the log file. If not provided, logs will be printed to the console only.
        log_level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(json_file))
        

def load_json(filename):
    with open(filename) as f:
        json_file = json.loads(f.read())
    return json_file

def print_study(study, logger):
    logger.info(f'Number of finished trials: {len(study.trials)} ')
    logger.info('Best trial:')
    trial = study.best_trial
    logger.info(f'  Value: {trial.value}')
    logger.info(f'  Params: ')
    for key, value in trial.params.items():
        logger.info(f'    {key}: {value}')