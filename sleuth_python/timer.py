""" Defines a timer class,
based on https://realpython.com/python-timer/"""

import time
from collections import namedtuple


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    timers = dict()

    def __init__(self,
                 name=None,
                 text="Elapsed time: {:0.4f} seconds",
                 logger=print):
        self._start_time = None
        self.name = name
        self.text = text
        self.logger = logger

        # Add new named timers to dictionary of timers
        if name:
            self.timers.setdefault(name, 0)

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. "
                             "Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. "
                             "Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time


timer_names = [
    "SPR_TOTAL",
    "SPR_PHASE1N3",
    "SPR_PHASE4",
    "SPR_PHASE5",
    "GDIF_WRITEGIF",
    "GDIF_READGIF",
    "DELTA_DELTATRON",
    "DELTA_PHASE1",
    "DELTA_PHASE2",
    "GROW",
    "DRIVER",
    "TOTAL_TIME"
]

Timers = namedtuple("Timers", " ".join(timer_names))

timers = Timers(*[Timer(name=name,
                        text=f"{name} Elapsed time: {{:0.4f}} seconds",
                        logger=print) for name in timer_names])
