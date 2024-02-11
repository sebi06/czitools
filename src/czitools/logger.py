import logging
import datetime


class CustomFormatter(logging.Formatter):
    """
    Logging colored formatter.
    This class is used to colorize the output of a logger based on the level of the log message.

    Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt

        # define ANSI color codes - see: https://talyian.github.io/ansicolors/
        grey = "\x1b[38;21m"
        green = "\x1b[0;30;42m"
        yellow = "\x1b[1;30;43m"
        orange = "\x1b[38;5;214m"
        red = "\x1b[0:30;41m"
        bold_red = "\x1b[1;30;1m"
        reset = "\x1b[0m"

        self.FORMATS = {
            logging.DEBUG: f"%(asctime)s -  {grey}%(levelname)s{reset} - %(message)s",
            logging.INFO: f"%(asctime)s -  {green}%(levelname)s{reset} - %(message)s",
            logging.WARNING: f"%(asctime)s -  {orange}%(levelname)s{reset} - %(message)s",
            logging.ERROR: f"%(asctime)s -  {red}%(levelname)s{reset} - %(message)s",
            logging.CRITICAL: f"%(asctime)s -  {bold_red}%(levelname)s{reset} - %(message)s",
        }

    def format(self, record):
        """
        Format the specified record as text.

        :param record: A LogRecord class object.
        :return: Returns the formatted record.
        """

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(log_to_file: bool = False):
    """
    Creates a custom logger with two handlers:
    - stdout_handler: logging to console (logs all five levels)
    - file_handler: logging to a file (logs all five levels) if `log_to_file` is True

    Args:
        log_to_file (bool): Whether or not to log to file. Defaults to False.

    Returns:
        logging.Logger: A custom logger with two handlers.
    """

    # Create custom logger logging all five levels
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Define format for logs
    fmt = "%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)2d | %(message)s"

    # Create stdout handler for logging to the console (logs all five levels)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(CustomFormatter(fmt))

    # Create file handler for logging to a file (logs all five levels)
    today = datetime.date.today()

    # make sure to only create it once otherwise one will get double entries in log
    if not logger.hasHandlers():
        # Add both handlers to the logger
        logger.addHandler(stdout_handler)

        if log_to_file:
            file_handler = logging.FileHandler(
                "my_app_{}.log".format(today.strftime("%Y_%m_%d"))
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(file_handler)

    return logger
