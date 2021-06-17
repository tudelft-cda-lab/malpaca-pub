import logging
import time
from typing import TypeVar, Callable

T = TypeVar('T')


def timeFunction(name, fun: Callable[[], T]) -> T:
    logging.debug(f"Started {name}...")
    startTime = time.perf_counter()
    value = fun()
    endTime = time.perf_counter()
    logging.info(f"Completed {name} in {endTime - startTime:0.4f} seconds")
    return value
