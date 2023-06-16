import logging

from rich.logging import RichHandler

# from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.addHandler(RichHandler())
    # file_formatter = logging.Formatter(fmt_file)
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # logging_levels = (
    #     "debug",
    #     "info",
    #     "warning",
    #     "error",
    #     "exception",
    #     "fatal",
    #     "critical",
    # )
    # for level in logging_levels:
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
