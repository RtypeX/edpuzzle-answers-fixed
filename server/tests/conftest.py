import sys
import os
import json
import pathlib
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup: make `server/` importable as the top-level package directory.
# ---------------------------------------------------------------------------
server_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, server_dir)

# ---------------------------------------------------------------------------
# Minimal test configuration – mirrors default.json but without real creds.
# ---------------------------------------------------------------------------
TEST_CONFIG = {
    "enable_auto_answer": True,
    "dev_mode": False,
    "include_traceback": False,
    "behind_proxy": False,
    "gzip_responses": False,
    "server_port": 5001,
    "limiter_storage_uri": "memory://",
    "script.js": "http://localhost:5001/script.js",
    "teacher_creds": [
        {"username": "test@example.com", "password": "testpass"}
    ],
    "gemini": {
        "key": "test-key",
        "model": "test-model",
    },
    "rate_limit": {
        "captions": "1000/minute",
        "generate": "1000/minute",
        "media": "1000/minute",
    },
}

# ---------------------------------------------------------------------------
# Patch pathlib.Path so that main.py can be imported without a real config
# file on disk.  Only the two specific paths used by main.py at module level
# are intercepted; everything else falls through to the real implementation.
# ---------------------------------------------------------------------------
_real_read_text = pathlib.Path.read_text
_real_exists = pathlib.Path.exists
_real_mkdir = pathlib.Path.mkdir


def _mock_read_text(self, *args, **kwargs):
    if self.name == "config.json":
        return json.dumps(TEST_CONFIG)
    return _real_read_text(self, *args, **kwargs)


def _mock_exists(self):
    # Pretend there is no cache file so main.py skips cache loading.
    if self.name == "cache.json":
        return False
    return _real_exists(self)


def _mock_mkdir(self, *args, **kwargs):
    pass


with (
    patch.object(pathlib.Path, "read_text", _mock_read_text),
    patch.object(pathlib.Path, "exists", _mock_exists),
    patch.object(pathlib.Path, "mkdir", _mock_mkdir),
):
    import main as _main_module  # noqa: E402 – intentional late import

# ---------------------------------------------------------------------------
# Shared pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Return the Flask application configured for testing."""
    _main_module.app.config.update(
        {
            "TESTING": True,
            "RATELIMIT_ENABLED": False,  # disable Flask-Limiter in tests
        }
    )
    return _main_module.app


@pytest.fixture
def client(app):
    """Return a Flask test client."""
    with app.test_client() as c:
        yield c
