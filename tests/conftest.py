import faulthandler
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_TIMEOUT_SECONDS = 180


@pytest.fixture(autouse=True)
def _hang_guard():
    """A hung test dumps every thread's stack and ends the run instead of blocking silently.

    (One unattended run once took 1 h 46 min; this makes such an event loud and diagnosable.)"""
    faulthandler.dump_traceback_later(TEST_TIMEOUT_SECONDS, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()
