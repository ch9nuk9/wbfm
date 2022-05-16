import logging
from pathlib import Path


def setup_logger(log_filename):
    # Log to a file and the console
    # https://docs.python.org/3/howto/logging-cookbook.html
    log_name = Path(log_filename.stem)
    logger = logging.getLogger(f'{log_name}')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_filename,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the base logger
    logger.addHandler(console)
    return logger
