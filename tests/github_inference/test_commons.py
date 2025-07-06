"""
Tests for the commons module.

Author: Ron Webb
Since: 1.0.0
"""

import time
from unittest import mock
from unittest.mock import Mock, patch
import requests
from sample.github_inference.rate_limiter import RateLimiter


class TestRateLimiter:
    """
    Test class for RateLimiter.
    
    Author: Ron Webb
    Since: 1.0.0
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter()
        self.test_url = "https://api.github.com/test"
        self.test_headers = {"Authorization": "Bearer test-token"}
        self.test_payload = {"test": "data"}

    @patch('requests.post')
    def test_make_request_success(self, mock_post):
        """Test successful HTTP request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.rate_limiter.make_request(
            self.test_url, self.test_headers, self.test_payload
        )
        
        assert result == mock_response
        mock_post.assert_called_once_with(
            self.test_url, headers=self.test_headers, json=self.test_payload, timeout=30
        )

    @patch('requests.post')
    @patch('time.sleep')
    def test_make_request_rate_limited_with_retry_after(self, mock_sleep, mock_post):
        """Test HTTP request with rate limiting and retry-after header."""
        # First call returns 429, second call succeeds
        rate_limited_response = Mock()
        rate_limited_response.status_code = 429
        rate_limited_response.headers = {'retry-after': '5'}
        rate_limited_response.raise_for_status.side_effect = None
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None
        
        mock_post.side_effect = [rate_limited_response, success_response]
        
        result = self.rate_limiter.make_request(
            self.test_url, self.test_headers, self.test_payload
        )
        
        assert result == success_response
        assert mock_post.call_count == 2
        mock_sleep.assert_called_with(5)

    @patch('requests.post')
    @patch('time.sleep')
    def test_make_request_rate_limited_with_exponential_backoff(self, mock_sleep, mock_post):
        """Test HTTP request with rate limiting and exponential backoff."""
        # First call returns 429, second call succeeds
        rate_limited_response = Mock()
        rate_limited_response.status_code = 429
        rate_limited_response.headers = {}
        rate_limited_response.raise_for_status.side_effect = None
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}
        success_response.raise_for_status.return_value = None
        
        mock_post.side_effect = [rate_limited_response, success_response]
        
        result = self.rate_limiter.make_request(
            self.test_url, self.test_headers, self.test_payload
        )
        
        assert result == success_response
        assert mock_post.call_count == 2
        mock_sleep.assert_called_with(1.0)  # base_delay * (2 ** 0)

    @patch('requests.post')
    def test_make_request_max_retries_exceeded(self, mock_post):
        """Test HTTP request exceeding maximum retries."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Rate limited")
        
        mock_post.return_value = mock_response
        
        try:
            self.rate_limiter.make_request(
                self.test_url, self.test_headers, self.test_payload, max_retries=2
            )
            assert False, "Expected HTTPError to be raised"
        except requests.exceptions.HTTPError:
            pass  # Expected
        
        assert mock_post.call_count == 3  # Initial + 2 retries

    @patch('requests.post')
    def test_make_request_request_exception(self, mock_post):
        """Test HTTP request with request exception."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        try:
            self.rate_limiter.make_request(
                self.test_url, self.test_headers, self.test_payload, max_retries=1
            )
            assert False, "Expected RequestException to be raised"
        except requests.exceptions.RequestException:
            pass  # Expected
        
        assert mock_post.call_count == 2  # Initial + 1 retry

    def test_rate_limiter_initialization(self):
        """Test RateLimiter can be initialized."""
        rate_limiter = RateLimiter()
        assert rate_limiter is not None

    @patch('requests.post')
    def test_make_request_with_custom_timeout(self, mock_post):
        """Test make_request with custom timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.rate_limiter.make_request(
            self.test_url, self.test_headers, self.test_payload, timeout=60
        )
        
        assert result == mock_response
        mock_post.assert_called_once_with(
            self.test_url, headers=self.test_headers, json=self.test_payload, timeout=60
        )
