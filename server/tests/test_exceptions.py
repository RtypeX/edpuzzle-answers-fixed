"""Tests for server/modules/exceptions.py"""
import pytest

from modules.exceptions import (
    BadGatewayError,
    BadRequestError,
    ForbiddenError,
    ServiceUnavailableException,
    UnauthorizedError,
)


class TestExceptionStatusCodes:
    """Each custom exception must carry the correct HTTP status code."""

    def test_unauthorized_error_status_code(self):
        assert UnauthorizedError.status_code == 401

    def test_forbidden_error_status_code(self):
        assert ForbiddenError.status_code == 403

    def test_bad_request_error_status_code(self):
        assert BadRequestError.status_code == 400

    def test_bad_gateway_error_status_code(self):
        assert BadGatewayError.status_code == 502

    def test_service_unavailable_status_code(self):
        assert ServiceUnavailableException.status_code == 503


class TestExceptionInheritance:
    """All custom exceptions must derive from the built-in Exception."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            UnauthorizedError,
            ForbiddenError,
            BadRequestError,
            BadGatewayError,
            ServiceUnavailableException,
        ],
    )
    def test_inherits_from_exception(self, exc_class):
        assert issubclass(exc_class, Exception)


class TestExceptionRaiseAndCatch:
    """Custom exceptions must be raise-able and catch-able as expected."""

    def test_bad_request_error_message(self):
        with pytest.raises(BadRequestError, match="missing field"):
            raise BadRequestError("missing field")

    def test_unauthorized_error_message(self):
        with pytest.raises(UnauthorizedError, match="not authenticated"):
            raise UnauthorizedError("not authenticated")

    def test_forbidden_error_message(self):
        with pytest.raises(ForbiddenError, match="access denied"):
            raise ForbiddenError("access denied")

    def test_bad_gateway_error_message(self):
        with pytest.raises(BadGatewayError, match="upstream error"):
            raise BadGatewayError("upstream error")

    def test_service_unavailable_message(self):
        with pytest.raises(ServiceUnavailableException, match="offline"):
            raise ServiceUnavailableException("offline")

    def test_catch_as_generic_exception(self):
        """Custom exceptions must be catchable as plain Exception."""
        try:
            raise BadRequestError("oops")
        except Exception as e:
            assert str(e) == "oops"
        else:
            pytest.fail("Exception was not raised")
