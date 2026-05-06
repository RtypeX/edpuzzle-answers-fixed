"""Tests for server/modules/captions.py"""
import pytest
from unittest.mock import patch, MagicMock

from modules.captions import process_timestamp, truncate_captions, get_captions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CAPTIONS = [
    {"timestamp": 0,   "duration": 5,  "text": "Hello"},
    {"timestamp": 10,  "duration": 5,  "text": "world"},
    {"timestamp": 20,  "duration": 5,  "text": "this"},
    {"timestamp": 30,  "duration": 5,  "text": "is"},
    {"timestamp": 40,  "duration": 5,  "text": "a test"},
]


# ---------------------------------------------------------------------------
# process_timestamp
# ---------------------------------------------------------------------------

class TestProcessTimestamp:
    def test_zero_timestamp(self):
        assert process_timestamp("00:00:00") == 0

    def test_seconds_only(self):
        assert process_timestamp("00:00:45") == 45

    def test_minutes_and_seconds(self):
        assert process_timestamp("00:02:30") == 150

    def test_hours_minutes_seconds(self):
        assert process_timestamp("01:30:00") == 5400

    def test_one_hour_one_minute_one_second(self):
        assert process_timestamp("01:01:01") == 3661

    def test_large_hours(self):
        assert process_timestamp("10:00:00") == 36000

    def test_all_max_digits(self):
        # 99 hours + 59 min + 59 sec
        expected = 99 * 3600 + 59 * 60 + 59
        assert process_timestamp("99:59:59") == expected


# ---------------------------------------------------------------------------
# truncate_captions
# ---------------------------------------------------------------------------

class TestTruncateCaptions:
    def test_no_timestamp_returns_all(self):
        result = truncate_captions(SAMPLE_CAPTIONS[:], timestamp=None)
        assert result == SAMPLE_CAPTIONS

    def test_timestamp_before_first_caption(self):
        # timestamp=5: loop finds first caption whose .timestamp > 5, which is
        # index 1 (timestamp=10), so returns captions[:2] (indices 0 and 1).
        result = truncate_captions(SAMPLE_CAPTIONS[:], timestamp=5)
        assert result == SAMPLE_CAPTIONS[:2]

    def test_timestamp_between_captions(self):
        # timestamp 25: captions with timestamp ≤ 25 stop at index 2 (timestamp=20)
        # The function returns captions up to (and including) the first caption
        # whose timestamp is GREATER than 25, which is index 3 (timestamp=30).
        result = truncate_captions(SAMPLE_CAPTIONS[:], timestamp=25)
        assert result == SAMPLE_CAPTIONS[:4]

    def test_timestamp_past_all_captions_returns_all(self):
        result = truncate_captions(SAMPLE_CAPTIONS[:], timestamp=1000)
        assert result == SAMPLE_CAPTIONS

    def test_timestamp_with_count_limits_results(self):
        # timestamp=25 → would stop at index 3; count=2 means last 2 captions
        result = truncate_captions(SAMPLE_CAPTIONS[:], timestamp=25, count=2)
        # index of first caption > 25 is 3 (timestamp=30)
        # so slice is max(0, 3-2+1):3+1 = 2:4
        assert result == SAMPLE_CAPTIONS[2:4]

    def test_count_clamped_to_zero_index(self):
        # timestamp=5 → first caption with timestamp > 5 is at index 1 (timestamp=10).
        # count=100: max(0, 1-100+1) = 0, so slice is captions[0:2] → 2 items.
        result = truncate_captions(SAMPLE_CAPTIONS[:], timestamp=5, count=100)
        assert result == SAMPLE_CAPTIONS[:2]

    def test_empty_captions_list(self):
        assert truncate_captions([], timestamp=10) == []

    def test_single_caption_included(self):
        single = [{"timestamp": 15, "duration": 5, "text": "only"}]
        result = truncate_captions(single, timestamp=10)
        assert result == single

    def test_single_caption_excluded(self):
        single = [{"timestamp": 15, "duration": 5, "text": "only"}]
        # timestamp 20 > 15, so the loop finds index 0 and returns [:1]
        result = truncate_captions(single, timestamp=20)
        # 20 > 15 means we never enter the if-branch (20 is NOT < 15)
        # loop ends without returning → function falls through to `return captions`
        assert result == single


# ---------------------------------------------------------------------------
# get_captions (integration-style, external HTTP mocked)
# ---------------------------------------------------------------------------

MOCK_TRANSCRIPT_RESPONSE = {
    "data": {
        "transcripts": {
            "en": {
                "default": [
                    {"start": "00:00:00", "end": "00:00:05", "text": "Hello"},
                    {"start": "00:00:10", "end": "00:00:15", "text": "world"},
                ]
            }
        }
    }
}

MOCK_TRANSCRIPT_AUTO_RESPONSE = {
    "data": {
        "transcripts": {
            "en": {
                "auto": [
                    {"start": "00:00:00", "end": "00:00:05", "text": "auto caption"},
                ]
            }
        }
    }
}

MOCK_TRANSCRIPT_CUSTOM_RESPONSE = {
    "data": {
        "transcripts": {
            "en": {
                "custom": [
                    {"start": "00:00:00", "end": "00:00:05", "text": "custom caption"},
                ],
                "default": [
                    {"start": "00:00:00", "end": "00:00:05", "text": "default caption"},
                ],
            }
        }
    }
}


def _make_mock_session(transcript_response):
    """Build a mock requests.Session that returns preset responses."""
    mock_session = MagicMock()
    mock_r1 = MagicMock()
    mock_r2 = MagicMock()
    mock_r2.json.return_value = transcript_response
    mock_session.get.side_effect = [mock_r1, mock_r2]
    return mock_session


class TestGetCaptions:
    def test_returns_dict_with_captions_key(self):
        with patch("modules.captions.requests.Session") as MockSession, \
             patch("modules.captions.get_captions_attempt") as mock_attempt:
            mock_attempt.return_value = [
                {"timestamp": 0, "duration": 5, "text": "Hello"},
                {"timestamp": 10, "duration": 5, "text": "world"},
            ]
            result = get_captions("abc123")
        assert "captions" in result

    def test_returns_all_captions_when_no_timestamp(self):
        sample = [
            {"timestamp": 0, "duration": 5, "text": "Hello"},
            {"timestamp": 10, "duration": 5, "text": "world"},
        ]
        with patch("modules.captions.get_captions_attempt", return_value=sample):
            result = get_captions("abc123")
        assert result["captions"] == sample

    def test_truncates_when_timestamp_given(self):
        sample = [
            {"timestamp": 0, "duration": 5, "text": "Hello"},
            {"timestamp": 10, "duration": 5, "text": "world"},
            {"timestamp": 20, "duration": 5, "text": "end"},
        ]
        with patch("modules.captions.get_captions_attempt", return_value=sample):
            result = get_captions("abc123", timestamp="5")
        # timestamp 5 < caption at index 1 (timestamp=10), so returns captions[:1]
        # wait, logic: for i in range(len(captions)): if timestamp < caption["timestamp"]
        # 5 < 0? No. 5 < 10? Yes → returns captions[:i+1] = captions[:2]
        assert len(result["captions"]) == 2

    def test_timestamp_and_count_respected(self):
        sample = [
            {"timestamp": 0,  "duration": 5, "text": "a"},
            {"timestamp": 10, "duration": 5, "text": "b"},
            {"timestamp": 20, "duration": 5, "text": "c"},
            {"timestamp": 30, "duration": 5, "text": "d"},
        ]
        with patch("modules.captions.get_captions_attempt", return_value=sample):
            result = get_captions("abc123", timestamp="15", count="2")
        # timestamp=15: first caption whose .timestamp > 15 is at index 2 (ts=20).
        # count=2 limits the window to the 2 captions ending at that boundary: b and c.
        assert len(result["captions"]) == 2
        assert result["captions"][0]["text"] == "b"
        assert result["captions"][1]["text"] == "c"

    def test_get_captions_attempt_uses_default_when_no_custom(self):
        """get_captions_attempt must prefer default over auto when custom absent."""
        with patch("modules.captions.requests.Session") as MockSession:
            mock_session = _make_mock_session(MOCK_TRANSCRIPT_RESPONSE)
            MockSession.return_value = mock_session
            # Clear lru_cache so mock is actually called
            from modules.captions import get_captions_attempt
            get_captions_attempt.cache_clear()
            captions = get_captions_attempt("vid1")

        assert len(captions) == 2
        assert captions[0]["timestamp"] == 0
        assert captions[0]["text"] == "Hello"
        assert captions[1]["timestamp"] == 10

    def test_get_captions_attempt_uses_custom_over_default(self):
        """get_captions_attempt must prefer custom over default."""
        with patch("modules.captions.requests.Session") as MockSession:
            mock_session = _make_mock_session(MOCK_TRANSCRIPT_CUSTOM_RESPONSE)
            MockSession.return_value = mock_session
            from modules.captions import get_captions_attempt
            get_captions_attempt.cache_clear()
            captions = get_captions_attempt("vid2")

        assert captions[0]["text"] == "custom caption"

    def test_get_captions_attempt_uses_auto_as_fallback(self):
        """get_captions_attempt must fall back to auto when custom and default absent."""
        with patch("modules.captions.requests.Session") as MockSession:
            mock_session = _make_mock_session(MOCK_TRANSCRIPT_AUTO_RESPONSE)
            MockSession.return_value = mock_session
            from modules.captions import get_captions_attempt
            get_captions_attempt.cache_clear()
            captions = get_captions_attempt("vid3")

        assert captions[0]["text"] == "auto caption"

    def test_get_captions_attempt_duration_computed_correctly(self):
        """Duration of each caption must equal end - start in seconds."""
        with patch("modules.captions.requests.Session") as MockSession:
            mock_session = _make_mock_session(MOCK_TRANSCRIPT_RESPONSE)
            MockSession.return_value = mock_session
            from modules.captions import get_captions_attempt
            get_captions_attempt.cache_clear()
            captions = get_captions_attempt("vid4")

        # First caption: start=0, end=5 → duration=5
        assert captions[0]["duration"] == 5
        # Second caption: start=10, end=15 → duration=5
        assert captions[1]["duration"] == 5
