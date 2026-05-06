"""Tests for server/modules/utils.py"""
import pytest

import modules.utils as utils
from modules.exceptions import (
    BadGatewayError,
    BadRequestError,
    ForbiddenError,
    ServiceUnavailableException,
    UnauthorizedError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raise(exc):
    """Helper that raises exc so it gets a real traceback."""
    raise exc


# ---------------------------------------------------------------------------
# handle_exception – success path returns (dict, status_code)
# ---------------------------------------------------------------------------

class TestHandleExceptionResponseShape:
    """Verify the response dict always contains the required keys."""

    def test_returns_tuple(self):
        response, status = utils.handle_exception(BadRequestError("bad"))
        assert isinstance(response, dict)
        assert isinstance(status, int)

    def test_dict_contains_error_key(self):
        response, _ = utils.handle_exception(BadRequestError("x"))
        assert "error" in response

    def test_dict_contains_status_key(self):
        response, _ = utils.handle_exception(BadRequestError("x"))
        assert "status" in response

    def test_dict_contains_message_key(self):
        response, _ = utils.handle_exception(BadRequestError("x"))
        assert "message" in response


# ---------------------------------------------------------------------------
# handle_exception – custom exception types
# ---------------------------------------------------------------------------

class TestHandleExceptionStatusCodes:
    """Each custom exception type maps to its own HTTP status code."""

    @pytest.mark.parametrize(
        "exc_class, expected_status",
        [
            (BadRequestError, 400),
            (UnauthorizedError, 401),
            (ForbiddenError, 403),
            (BadGatewayError, 502),
            (ServiceUnavailableException, 503),
        ],
    )
    def test_known_exception_status(self, exc_class, expected_status):
        response, status = utils.handle_exception(exc_class("msg"))
        assert status == expected_status
        assert response["status"] == expected_status

    def test_generic_exception_defaults_to_500(self):
        _, status = utils.handle_exception(RuntimeError("boom"))
        assert status == 500

    def test_exception_type_name_in_error_field(self):
        response, _ = utils.handle_exception(BadRequestError("test"))
        assert response["error"] == "BadRequestError"

    def test_message_matches_exception_message(self):
        response, _ = utils.handle_exception(BadRequestError("test message"))
        assert response["message"] == "test message"


# ---------------------------------------------------------------------------
# handle_exception – explicit status_code override
# ---------------------------------------------------------------------------

class TestHandleExceptionStatusCodeOverride:
    def test_override_takes_precedence_over_exception_status(self):
        _, status = utils.handle_exception(BadRequestError("x"), status_code=429)
        assert status == 429

    def test_override_reflected_in_response_dict(self):
        response, _ = utils.handle_exception(BadRequestError("x"), status_code=429)
        assert response["status"] == 429


# ---------------------------------------------------------------------------
# handle_exception – traceback inclusion
# ---------------------------------------------------------------------------

class TestHandleExceptionTraceback:
    def test_traceback_absent_by_default(self):
        utils.include_traceback = False
        response, _ = utils.handle_exception(BadRequestError("err"))
        assert "traceback" not in response

    def test_traceback_present_when_enabled(self):
        utils.include_traceback = True
        try:
            _raise(BadRequestError("err"))
        except BadRequestError as exc:
            response, _ = utils.handle_exception(exc)
        finally:
            utils.include_traceback = False
        assert "traceback" in response

    def test_traceback_is_string(self):
        utils.include_traceback = True
        try:
            _raise(BadRequestError("err"))
        except BadRequestError as exc:
            response, _ = utils.handle_exception(exc)
        finally:
            utils.include_traceback = False
        assert isinstance(response["traceback"], str)


# ---------------------------------------------------------------------------
# handle_exception – non-Exception argument (fallback branch)
# ---------------------------------------------------------------------------

class TestHandleExceptionNonException:
    def test_non_exception_returns_unknown_error(self):
        response, status = utils.handle_exception("not an exception")
        assert response["error"] == "Unknown"
        assert status == 500

    def test_none_returns_unknown_error(self):
        response, status = utils.handle_exception(None)
        assert response["error"] == "Unknown"
        assert status == 500
