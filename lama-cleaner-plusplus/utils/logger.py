import logging
import sys


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("core").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("ui").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("utils").setLevel(getattr(logging, level.upper(), logging.INFO))
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
