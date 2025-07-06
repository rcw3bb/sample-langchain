"""
Tests for configuration handler module.

Author: Ron Webb
Since: 1.0.0
"""

import pytest
from sample.github_inference.configuration_handler import ConfigurationHandler


class TestConfigurationHandler:
    """
    Test cases for ConfigurationHandler class.

    Author: Ron Webb
    Since: 1.0.0
    """

    def test_build_headers_success(self):
        """
        Test successful header building with valid API key.
        """
        api_key = "test_api_key_123"
        headers = ConfigurationHandler.build_headers(api_key)

        expected_headers = {
            "Authorization": "Bearer test_api_key_123",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        assert headers == expected_headers

    def test_build_request_payload_minimal(self):
        """
        Test building request payload with only required parameters.
        """
        model = "gpt-3.5-turbo"
        api_messages = [{"role": "user", "content": "Hello"}]

        payload = ConfigurationHandler.build_request_payload(model, api_messages)

        expected_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        assert payload == expected_payload

    def test_build_request_payload_with_temperature(self):
        """
        Test building request payload with temperature parameter.
        """
        model = "gpt-3.5-turbo"
        api_messages = [{"role": "user", "content": "Hello"}]
        temperature = 0.7

        payload = ConfigurationHandler.build_request_payload(
            model, api_messages, temperature=temperature
        )

        expected_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.7,
        }

        assert payload == expected_payload

    def test_build_request_payload_with_max_tokens_none(self):
        """
        Test building request payload when max_tokens is None.
        """
        model = "gpt-3.5-turbo"
        api_messages = [{"role": "user", "content": "Hello"}]
        max_tokens = None

        payload = ConfigurationHandler.build_request_payload(
            model, api_messages, max_tokens=max_tokens
        )

        expected_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        assert payload == expected_payload
        assert "max_tokens" not in payload

    def test_build_request_payload_with_max_tokens_value(self):
        """
        Test building request payload when max_tokens has a value.
        """
        model = "gpt-3.5-turbo"
        api_messages = [{"role": "user", "content": "Hello"}]
        max_tokens = 100

        payload = ConfigurationHandler.build_request_payload(
            model, api_messages, max_tokens=max_tokens
        )

        expected_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 100,
        }

        assert payload == expected_payload

    def test_build_request_payload_with_max_tokens_zero(self):
        """
        Test building request payload when max_tokens is zero.
        """
        model = "gpt-3.5-turbo"
        api_messages = [{"role": "user", "content": "Hello"}]
        max_tokens = 0

        payload = ConfigurationHandler.build_request_payload(
            model, api_messages, max_tokens=max_tokens
        )

        expected_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 0,
        }

        assert payload == expected_payload

    def test_build_request_payload_with_all_parameters(self):
        """
        Test building request payload with all optional parameters.
        """
        model = "gpt-4"
        api_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        temperature = 0.8
        max_tokens = 150

        payload = ConfigurationHandler.build_request_payload(
            model, api_messages, temperature=temperature, max_tokens=max_tokens
        )

        expected_payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
            "temperature": 0.8,
            "max_tokens": 150,
        }

        assert payload == expected_payload

    def test_build_request_payload_temperature_none_max_tokens_value(self):
        """
        Test building request payload with temperature None and max_tokens with value.
        """
        model = "gpt-3.5-turbo"
        api_messages = [{"role": "user", "content": "Test"}]
        temperature = None
        max_tokens = 200

        payload = ConfigurationHandler.build_request_payload(
            model, api_messages, temperature=temperature, max_tokens=max_tokens
        )

        expected_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": False,
            "max_tokens": 200,
        }

        assert payload == expected_payload
        assert "temperature" not in payload

    def test_build_request_payload_with_large_max_tokens(self):
        """
        Test building request payload with a large max_tokens value.
        """
        model = "gpt-4"
        api_messages = [{"role": "user", "content": "Generate a long response"}]
        max_tokens = 4096

        payload = ConfigurationHandler.build_request_payload(
            model, api_messages, max_tokens=max_tokens
        )

        expected_payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Generate a long response"}],
            "stream": False,
            "max_tokens": 4096,
        }

        assert payload == expected_payload
