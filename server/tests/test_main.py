"""Tests for the Flask API routes defined in server/main.py"""
import json
from unittest.mock import MagicMock, patch

import pytest

from modules.exceptions import (
    BadGatewayError,
    BadRequestError,
    ServiceUnavailableException,
)


# ---------------------------------------------------------------------------
# /api/captions/<id>
# ---------------------------------------------------------------------------

class TestCaptionsRoute:
    def test_returns_200_with_captions(self, client):
        sample = [{"timestamp": 0, "duration": 5, "text": "Hello"}]
        with patch("modules.captions.get_captions", return_value={"captions": sample}):
            resp = client.get("/api/captions/abc123")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "captions" in data

    def test_exception_returns_error_json(self, client):
        with patch(
            "modules.captions.get_captions",
            side_effect=RuntimeError("fetch failed"),
        ):
            resp = client.get("/api/captions/bad_id")
        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert data["error"] == "RuntimeError"

    def test_bad_request_exception_returns_400(self, client):
        with patch(
            "modules.captions.get_captions",
            side_effect=BadRequestError("invalid id"),
        ):
            resp = client.get("/api/captions/bad_id")
        assert resp.status_code == 400

    def test_route_accepts_language_segment(self, client):
        sample = [{"timestamp": 0, "duration": 5, "text": "Hola"}]
        with patch("modules.captions.get_captions", return_value={"captions": sample}):
            resp = client.get("/api/captions/abc123/es")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------

class TestModelsRoute:
    def test_returns_200(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200

    def test_returns_models_key(self, client):
        resp = client.get("/api/models")
        data = json.loads(resp.data)
        assert "models" in data


# ---------------------------------------------------------------------------
# /api/generate  (POST)
# ---------------------------------------------------------------------------

class TestGenerateRoute:
    def _post(self, client, payload):
        return client.post(
            "/api/generate",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_missing_prompt_returns_400(self, client):
        resp = self._post(client, {"model": "m"})
        assert resp.status_code == 400
        assert b"prompt" in resp.data

    def test_missing_model_returns_400(self, client):
        resp = self._post(client, {"prompt": "hello"})
        assert resp.status_code == 400
        assert b"model" in resp.data

    def test_unknown_parameter_returns_400(self, client):
        resp = self._post(client, {"prompt": "hi", "model": "m", "extra": "x"})
        assert resp.status_code == 400

    def test_prompt_too_long_returns_400(self, client):
        import main as m
        original = m.ai.max_length
        m.ai.max_length = 5
        try:
            resp = self._post(client, {"prompt": "this is too long", "model": "m"})
        finally:
            m.ai.max_length = original
        assert resp.status_code == 400

    def test_valid_request_streams_response(self, client):
        def fake_generate(data):
            yield {"status": "generating"}
            yield {"text": "Hello"}
            yield {"status": "done"}

        with patch("main.ai.generate", side_effect=fake_generate):
            resp = self._post(client, {"prompt": "hi", "model": "test-model"})
        assert resp.status_code == 200
        # Response is newline-delimited JSON
        lines = [l for l in resp.data.decode().strip().split("\n") if l]
        events = [json.loads(l) for l in lines]
        statuses = [e.get("status") for e in events if "status" in e]
        assert "generating" in statuses
        assert "done" in statuses

    def test_generate_exception_returns_error_in_stream(self, client):
        def fake_generate(data):
            raise ServiceUnavailableException("AI offline")
            yield  # make it a generator

        with patch("main.ai.generate", side_effect=fake_generate):
            resp = self._post(client, {"prompt": "hi", "model": "test-model"})
        # The generator catches exceptions and yields a JSON error line
        assert resp.status_code == 200
        lines = [l for l in resp.data.decode().strip().split("\n") if l]
        events = [json.loads(l) for l in lines]
        error_events = [e for e in events if e.get("status") == "error"]
        assert error_events, "Expected at least one error event in the stream"


# ---------------------------------------------------------------------------
# /api/media/<media_id>
# ---------------------------------------------------------------------------

class TestMediaRoute:
    def _make_session(self, status_code=200, json_data=None):
        """Return a mock requests.Session whose .get() returns a preset response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.ok = status_code == 200
        mock_resp.json.return_value = json_data or {"questions": [], "source": "youtube"}
        mock_session = MagicMock()
        # session.get("csrf") → returns CSRF response; second .get → media response
        csrf_mock = MagicMock()
        csrf_mock.json.return_value = {"CSRFToken": "test-token"}
        mock_session.get.side_effect = [csrf_mock, mock_resp]
        return mock_session

    def test_returns_200_with_valid_media(self, client):
        mock_session = self._make_session(200, {"source": "youtube", "questions": []})
        with patch("main.create_session", return_value=mock_session), \
             patch("main.ensure_teacher_token", return_value=("tok", 9999)):
            resp = client.get("/api/media/media123")
        assert resp.status_code == 200

    def test_returns_data_from_edpuzzle(self, client):
        payload = {"source": "youtube", "questions": [{"_id": "q1"}]}
        mock_session = self._make_session(200, payload)
        with patch("main.create_session", return_value=mock_session), \
             patch("main.ensure_teacher_token", return_value=("tok", 9999)):
            resp = client.get("/api/media/media123")
        data = json.loads(resp.data)
        assert data["source"] == "youtube"

    def test_403_from_edpuzzle_returns_502(self, client):
        mock_session = self._make_session(403)
        with patch("main.create_session", return_value=mock_session), \
             patch("main.ensure_teacher_token", return_value=("tok", 9999)):
            resp = client.get("/api/media/private_media")
        assert resp.status_code == 502

    def test_non_200_from_edpuzzle_returns_502(self, client):
        mock_session = self._make_session(500)
        with patch("main.create_session", return_value=mock_session), \
             patch("main.ensure_teacher_token", return_value=("tok", 9999)):
            resp = client.get("/api/media/broken_media")
        assert resp.status_code == 502

    def test_edpuzzle_error_field_returns_502(self, client):
        mock_session = self._make_session(200, {"error": "not found"})
        with patch("main.create_session", return_value=mock_session), \
             patch("main.ensure_teacher_token", return_value=("tok", 9999)):
            resp = client.get("/api/media/err_media")
        assert resp.status_code == 502

    def test_auto_answer_disabled_returns_503(self, client):
        import main as m
        original = m.enable_auto_answer
        m.enable_auto_answer = False
        try:
            resp = client.get("/api/media/any")
        finally:
            m.enable_auto_answer = original
        assert resp.status_code == 503

    def test_no_configured_creds_returns_503(self, client):
        with patch(
            "main.ensure_teacher_token",
            side_effect=ServiceUnavailableException("no creds"),
        ):
            resp = client.get("/api/media/any")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# / (homepage)
# ---------------------------------------------------------------------------

class TestHomepageRoute:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, client):
        resp = client.get("/")
        assert b"html" in resp.data.lower()


# ---------------------------------------------------------------------------
# /discord
# ---------------------------------------------------------------------------

class TestDiscordRoute:
    def test_discord_redirects(self, client):
        resp = client.get("/discord")
        assert resp.status_code in (301, 302)

    def test_discord_html_redirects(self, client):
        resp = client.get("/discord.html")
        assert resp.status_code in (301, 302)

    def test_redirect_points_to_discord(self, client):
        resp = client.get("/discord")
        location = resp.headers.get("Location", "")
        assert location.startswith("https://discord.com/")


# ---------------------------------------------------------------------------
# Rate-limit error handler (429)
# ---------------------------------------------------------------------------

class TestRateLimitHandler:
    def test_429_handler_returns_json(self, client):
        """The custom 429 handler must return a JSON body with status 429."""
        import main as m
        with m.app.test_request_context():
            exc = RuntimeError("rate limit exceeded")
            body, status = m.handle_rate_limit(exc)
        assert status == 429
        assert "error" in body


# ---------------------------------------------------------------------------
# Utility functions (main.py helpers)
# ---------------------------------------------------------------------------

class TestIsConfiguredCredential:
    def test_valid_credentials_pass(self):
        import main as m
        assert m.is_configured_credential({"username": "a@b.com", "password": "secret"})

    def test_placeholder_username_fails(self):
        import main as m
        assert not m.is_configured_credential(
            {"username": "EDPUZZLE_EMAIL", "password": "pass"}
        )

    def test_placeholder_password_fails(self):
        import main as m
        assert not m.is_configured_credential(
            {"username": "user@example.com", "password": "EDPUZZLE_PASSWORD"}
        )

    def test_empty_username_fails(self):
        import main as m
        assert not m.is_configured_credential({"username": "", "password": "pass"})

    def test_empty_password_fails(self):
        import main as m
        assert not m.is_configured_credential({"username": "user", "password": ""})

    def test_whitespace_only_credentials_fail(self):
        import main as m
        assert not m.is_configured_credential(
            {"username": "  ", "password": "  "}
        )


class TestGetConfiguredTeacherCreds:
    def test_filters_out_placeholder_creds(self):
        import main as m
        original = m.config["teacher_creds"]
        m.config["teacher_creds"] = [
            {"username": "EDPUZZLE_EMAIL", "password": "EDPUZZLE_PASSWORD"},
            {"username": "real@example.com", "password": "realpass"},
        ]
        try:
            result = m.get_configured_teacher_creds()
        finally:
            m.config["teacher_creds"] = original
        assert len(result) == 1
        assert result[0]["username"] == "real@example.com"

    def test_empty_list_when_all_placeholders(self):
        import main as m
        original = m.config["teacher_creds"]
        m.config["teacher_creds"] = [
            {"username": "EDPUZZLE_EMAIL", "password": "EDPUZZLE_PASSWORD"}
        ]
        try:
            result = m.get_configured_teacher_creds()
        finally:
            m.config["teacher_creds"] = original
        assert result == []


# ---------------------------------------------------------------------------
# ensure_teacher_token – unit tests
# ---------------------------------------------------------------------------

class TestEnsureTeacherToken:
    def _with_tokens(self, m, tokens_dict):
        """Context manager that temporarily replaces current_tokens."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            original = dict(m.current_tokens)
            m.current_tokens.clear()
            m.current_tokens.update(tokens_dict)
            try:
                yield
            finally:
                m.current_tokens.clear()
                m.current_tokens.update(original)

        return _ctx()

    def test_returns_a_token_when_one_is_cached(self):
        import main as m
        entry = ("mytoken", int(__import__("time").time()))
        with self._with_tokens(m, {"user@example.com": entry}):
            result = m.ensure_teacher_token()
        assert result == entry

    def test_returns_one_of_many_cached_tokens(self):
        import main as m
        entries = {
            "a@example.com": ("tok_a", 1),
            "b@example.com": ("tok_b", 2),
        }
        with self._with_tokens(m, entries):
            result = m.ensure_teacher_token()
        assert result in entries.values()

    def test_no_cached_tokens_no_creds_raises_503(self):
        import main as m
        with self._with_tokens(m, {}):
            with patch("main.get_configured_teacher_creds", return_value=[]):
                with pytest.raises(Exception) as exc_info:
                    m.ensure_teacher_token()
        assert exc_info.value.__class__.__name__ == "ServiceUnavailableException"

    def test_no_cached_tokens_login_fails_raises_503(self):
        import main as m
        with self._with_tokens(m, {}):
            creds = [{"username": "u@example.com", "password": "pw"}]
            with patch("main.get_configured_teacher_creds", return_value=creds), \
                 patch("main.account_login", return_value=None):
                with pytest.raises(Exception) as exc_info:
                    m.ensure_teacher_token()
        assert exc_info.value.__class__.__name__ == "ServiceUnavailableException"

    def test_race_condition_tokens_emptied_before_choice(self):
        """Regression: current_tokens emptied between check and choice must not
        raise IndexError (i.e., random.choice on an empty sequence)."""
        import main as m

        # Simulate the race: patch current_tokens so that iterating .values()
        # returns an empty list even though the dict appeared non-empty.
        class _RacyDict(dict):
            def values(self):
                return {}.values()  # always returns empty view

        original = m.current_tokens
        # Inject a "non-empty" dict whose .values() behaves as if emptied.
        racy = _RacyDict({"user@example.com": ("tok", 1)})
        m.current_tokens = racy
        try:
            # The function must NOT raise IndexError; it falls through to the
            # credential / ServiceUnavailableException path instead.
            with patch("main.get_configured_teacher_creds", return_value=[]):
                with pytest.raises(Exception) as exc_info:
                    m.ensure_teacher_token()
            assert exc_info.value.__class__.__name__ == "ServiceUnavailableException"
        finally:
            m.current_tokens = original
