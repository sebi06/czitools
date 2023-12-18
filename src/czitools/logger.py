import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    green = "\x1b[0;30;42m"
    yellow = "\x1b[1;30;43m"
    red = "\x1b[0:30;41m"
    bold_red = "\x1b[1;30;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: f"%(asctime)s -  {grey}%(levelname)s{reset} - %(message)s",
        logging.INFO: f"%(asctime)s -  {green}%(levelname)s{reset} - %(message)s",
        logging.WARNING: f"%(asctime)s -  {yellow}%(levelname)s{reset} - %(message)s",
        logging.ERROR: f"%(asctime)s -  {red}%(levelname)s{reset} - %(message)s",
        logging.CRITICAL: f"%(asctime)s -  {bold_red}%(levelname)s{reset} - %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("core")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
