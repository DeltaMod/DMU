import logging

class CustomFormatter(logging.Formatter):
    magenta = "\033[1;35m"
    lblue = "\033[1;34m"
    yellow = "\033[1;33m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s \n %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: magenta + format + reset,
        logging.INFO: lblue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_custom_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    ch.setFormatter(CustomFormatter())
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(ch)

    return logger