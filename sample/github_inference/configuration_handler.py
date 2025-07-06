"""
Configuration handler for managing parameter validation and HTTP configuration.

Author: Ron Webb
Since: 1.0.0
"""

from typing import Any, Optional

# API endpoints and headers
GITHUB_API_ACCEPT_HEADER = "application/vnd.github+json"
GITHUB_API_CONTENT_TYPE = "application/json"
GITHUB_API_VERSION = "2022-11-28"


class ConfigurationHandler:
    """
    Handles parameter validation and HTTP configuration for the chat model.

    Author: Ron Webb
    Since: 1.0.0
    """

    @staticmethod
    def build_headers(api_key: str) -> dict[str, str]:
        """
        Build HTTP headers for API requests.

        Args:
            api_key: GitHub API key for authentication

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Authorization": f"Bearer {api_key}",
            "Accept": GITHUB_API_ACCEPT_HEADER,
            "Content-Type": GITHUB_API_CONTENT_TYPE,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }

    @staticmethod
    def build_request_payload(
        model: str,
        api_messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Build request payload for API calls.

        Args:
            model: Model name
            api_messages: Messages in API format
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Request payload dictionary
        """
        payload = {"model": model, "messages": api_messages, "stream": False}

        # Add optional parameters if specified
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return payload
