import sys
from functools import lru_cache
from pathlib import Path

from loguru import logger

PROJECT_PATH = str(Path(__file__).parents[1])
DATA_PATH = str(Path(PROJECT_PATH) / ".data")
Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
CONFIG_PATH = str(Path(PROJECT_PATH, "config"))

logger_format = "<green>{time:MM/DD HH:mm:ss.SS}</green> | <level>{level: ^7}</level> |" \
                "{process.name:<5}.{thread.name:<5}: " \
                "<cyan>{name:<8}</cyan>:<cyan>{function:<10}</cyan>:<cyan>{line:<4}</cyan>" \
                " - <level>{message}</level>"


@lru_cache()
def config_logger():
    logger.remove()

    logger.add(sys.stderr, format=logger_format, backtrace=False, diagnose=False, colorize=True)


config_logger()
