import logging
from pathlib import Path

from DLC_for_WBFM.utils.projects.utils_filenames import get_sequential_filename


def setup_logger(log_filename, filemode='w'):
    # Log to a file and the console
    # https://docs.python.org/3/howto/logging-cookbook.html
    log_name = Path(log_filename).stem

    # Main log object
    logger = logging.getLogger(f'{log_name}')

    # Set up file handler
    log_filename = get_sequential_filename(log_filename)
    print(f"Setting up log at: {log_filename}")
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                  datefmt='%m-%d %H:%M')
    fh.setFormatter(formatter)

    # Set up console handler (simplified compared to the file)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    ch.setFormatter(formatter)

    # Actually add the handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
