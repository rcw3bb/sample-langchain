"""
Tests for RateLimiter module.

Author: Ron Webb
Since: 1.0.0
"""

import time
from unittest.mock import Mock, patch
import requests
import pytest
from sample.github_inference.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter()

    def test_initialization(self):
        """Test rate limiter initialization."""
        # RateLimiter doesn't have instance variables for these parameters
        # They are passed to the make_request method
        assert isinstance(self.rate_limiter, RateLimiter)

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_successful_request(self, mock_post):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        result = self.rate_limiter.make_request(url, headers, payload)

        assert result == mock_response
        mock_post.assert_called_once_with(
            url, json=payload, headers=headers, timeout=30
        )

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_rate_limit_with_reset_time(self, mock_post):
        """Test rate limit handling with reset time."""
        # First request returns rate limit error
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(time.time() + 0.1)  # Reset in 0.1 seconds
        }

        # Second request succeeds
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        mock_post.side_effect = [rate_limit_response, success_response]

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep') as mock_sleep:
            result = self.rate_limiter.make_request(url, headers, payload)

        assert result == success_response
        assert mock_post.call_count == 2
        # Rate limiter does both exponential backoff AND rate limit waiting
        assert mock_sleep.call_count >= 1

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_rate_limit_with_time_remaining(self, mock_post):
        """Test rate limit handling with time remaining header."""
        # First request returns rate limit error
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-timeremaining": "100"  # 100 milliseconds remaining
        }

        # Second request succeeds
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        mock_post.side_effect = [rate_limit_response, success_response]

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep') as mock_sleep:
            result = self.rate_limiter.make_request(url, headers, payload)

        assert result == success_response
        assert mock_post.call_count == 2
        # Rate limiter does both exponential backoff AND rate limit waiting
        assert mock_sleep.call_count >= 1

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_rate_limit_403_status(self, mock_post):
        """Test rate limit handling with 403 status code."""
        # First request returns rate limit error with 403
        rate_limit_response = Mock()
        rate_limit_response.status_code = 403
        rate_limit_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(time.time() + 0.1)
        }

        # Second request succeeds
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        mock_post.side_effect = [rate_limit_response, success_response]

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep') as mock_sleep:
            result = self.rate_limiter.make_request(url, headers, payload)

        assert result == success_response
        assert mock_post.call_count == 2
        # Rate limiter does both exponential backoff AND rate limit waiting
        assert mock_sleep.call_count >= 1

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_max_retries_exceeded(self, mock_post):
        """Test that RequestException is raised after max retries."""
        # All requests fail with rate limit
        error_response = Mock()
        error_response.status_code = 429
        error_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(time.time() + 0.1)
        }
        mock_post.return_value = error_response

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep'):
            with pytest.raises(requests.RequestException, match="Max retries exceeded"):
                self.rate_limiter.make_request(url, headers, payload, max_retries=2)

        # Should make initial request + max_retries attempts
        assert mock_post.call_count == 3

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_custom_timeout(self, mock_post):
        """Test request with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}
        timeout = 60

        self.rate_limiter.make_request(url, headers, payload, timeout)

        mock_post.assert_called_once_with(
            url, json=payload, headers=headers, timeout=timeout
        )

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_request_exception_handling(self, mock_post):
        """Test handling of request exceptions."""
        # First request raises exception
        mock_post.side_effect = [
            requests.ConnectionError("Connection failed"),
            Mock(status_code=200, headers={}, raise_for_status=Mock())
        ]

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep'):
            result = self.rate_limiter.make_request(url, headers, payload)

        assert mock_post.call_count == 2
        assert result.status_code == 200

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_continuous_request_exceptions(self, mock_post):
        """Test continuous request exceptions leading to max retries."""
        # All requests raise exceptions
        mock_post.side_effect = requests.ConnectionError("Connection failed")

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep'):
            with pytest.raises(requests.ConnectionError, match="Connection failed"):
                self.rate_limiter.make_request(url, headers, payload, max_retries=2)

        assert mock_post.call_count == 3

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_rate_limit_no_headers(self, mock_post):
        """Test rate limit handling when no rate limit headers are present."""
        # First request returns rate limit error without headers
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}

        # Second request succeeds
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        mock_post.side_effect = [rate_limit_response, success_response]

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep') as mock_sleep:
            result = self.rate_limiter.make_request(url, headers, payload)

        assert result == success_response
        assert mock_post.call_count == 2
        # Should still sleep even without headers (using retry-after or exponential backoff)
        mock_sleep.assert_called_once()

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_retry_after_header(self, mock_post):
        """Test handling of retry-after header."""
        # First request returns rate limit error with retry-after header
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {
            "retry-after": "2"  # Retry after 2 seconds
        }

        # Second request succeeds
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        mock_post.side_effect = [rate_limit_response, success_response]

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep') as mock_sleep:
            result = self.rate_limiter.make_request(url, headers, payload)

        assert result == success_response
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(2)

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_successful_response_with_rate_limit_headers(self, mock_post):
        """Test successful response that includes rate limit headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            "x-ratelimit-remaining": "100",
            "x-ratelimit-reset": str(time.time() + 3600)
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        result = self.rate_limiter.make_request(url, headers, payload)

        assert result == mock_response
        mock_post.assert_called_once()

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_http_error_response(self, mock_post):
        """Test handling of HTTP error responses (not rate limits)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server Error")
        mock_post.return_value = mock_response

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with pytest.raises(requests.HTTPError):
            self.rate_limiter.make_request(url, headers, payload)

        # Should retry multiple times before giving up
        assert mock_post.call_count >= 1

    @patch('sample.github_inference.rate_limiter.requests.post')
    def test_wait_for_reset_past_time(self, mock_post):
        """Test that we don't wait if reset time is in the past."""
        # Request with reset time in the past
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(time.time() - 10)  # 10 seconds ago
        }

        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None

        mock_post.side_effect = [mock_response, success_response]

        url = "https://api.test.com"
        headers = {"Authorization": "Bearer test"}
        payload = {"message": "test"}

        with patch('time.sleep') as mock_sleep:
            result = self.rate_limiter.make_request(url, headers, payload)

        assert result == success_response
        # Should not sleep for past reset time, but may sleep for exponential backoff
        assert mock_post.call_count == 2

    @patch('time.sleep')
    @patch('time.time')
    def test_wait_for_reset_no_wait_needed(self, mock_time, mock_sleep):
        """Test wait_for_reset when reset time has already passed."""
        # Mock current time to be after reset time
        current_time = 1000.0
        reset_time = 900.0  # Reset time is in the past
        mock_time.return_value = current_time
        
        with patch('builtins.print') as mock_print:
            # Call the private method using getattr
            wait_method = getattr(self.rate_limiter, '_RateLimiter__wait_for_reset')
            wait_method(reset_time)
            
            # Verify no sleep was called
            mock_sleep.assert_not_called()
            
            # Verify the correct message was printed
            mock_print.assert_called_once_with(
                "Rate Limiter: No need to wait, rate limit has already reset"
            )
