"""
Keep Windows from sleeping or hibernating while a long job runs. Windows only
counts keyboard and mouse input as activity, so a multi-hour download or
training run can be put to sleep halfway (it hibernated after 45 idle minutes
on 2026-10-09). The request ends when the job does; no power setting changes.

    with keep_awake():
        long_job()
"""

import ctypes
import sys
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake():
    if sys.platform != "win32":
        yield
        return
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
