"""
Tool input parser for handling various tool input formats.

Author: Ron Webb
Since: 1.0.0
"""

import json
from typing import Any

# Constants for tool input parsing
DEFAULT_INPUT_KEY = "input"


class ToolInputParser:
    """
    Handles parsing of tool input with flexible format support.

    Author: Ron Webb
    Since: 1.0.0
    """

    def parse_tool_input(self, input_content: str) -> dict[str, Any]:
        """
        Parse tool input with flexible format support.

        Attempts to parse as JSON first, then falls back to intelligent parsing
        for common parameter formats.

        Args:
            input_content: Raw input content string

        Returns:
            Dictionary of parsed parameters
        """
        if not input_content:
            return {}

        input_content = input_content.strip()

        if not input_content:
            return {}

        json_result = self.__try_parse_json(input_content)
        if json_result is not None:
            return json_result

        try:
            return self.__parse_non_json_input(input_content)
        except Exception:
            return {DEFAULT_INPUT_KEY: input_content}

    def __try_parse_json(self, input_content: str) -> dict[str, Any] | None:
        """
        Attempt to parse input as JSON.

        Args:
            input_content: Raw input content string

        Returns:
            Parsed JSON as dictionary, or None if parsing fails
        """
        try:
            parsed = json.loads(input_content)
            if isinstance(parsed, str):
                return {DEFAULT_INPUT_KEY: parsed}
            return parsed
        except json.JSONDecodeError:
            return None

    def __parse_non_json_input(self, input_content: str) -> dict[str, Any]:
        """
        Parse non-JSON input content using various strategies.

        Args:
            input_content: Raw input content string

        Returns:
            Dictionary of parsed parameters
        """
        # Handle formats like: query="search term", limit=5
        if self.__is_key_value_format(input_content):
            return self.__parse_key_value_pairs(input_content)

        if self.__is_single_value_format(input_content):
            return self.__parse_single_value(input_content)

        return {DEFAULT_INPUT_KEY: input_content}

    def __is_key_value_format(self, input_content: str) -> bool:
        """
        Check if input content appears to be in key-value format.

        Args:
            input_content: Input content to check

        Returns:
            True if input appears to be key-value format
        """
        return "=" in input_content and (
            "," in input_content or input_content.count("=") == 1
        )

    def __is_single_value_format(self, input_content: str) -> bool:
        """
        Check if input content appears to be a single value.

        Args:
            input_content: Input content to check

        Returns:
            True if input appears to be a single value
        """
        return bool(
            input_content and not any(char in input_content for char in ["{", "[", "="])
        )

    def __parse_single_value(self, input_content: str) -> dict[str, str]:
        """
        Parse single value input, handling quoted strings.

        Args:
            input_content: Single value input content

        Returns:
            Dictionary with "input" key containing the parsed value
        """
        if input_content.startswith('"') and input_content.endswith('"'):
            return {DEFAULT_INPUT_KEY: input_content[1:-1]}
        elif input_content.startswith("'") and input_content.endswith("'"):
            return {DEFAULT_INPUT_KEY: input_content[1:-1]}
        else:
            return {DEFAULT_INPUT_KEY: input_content}

    def __parse_key_value_pairs(self, content: str) -> dict[str, Any]:
        """
        Parse key-value pairs from string content.

        Handles formats like:
        - query="search term", limit=5
        - name=value, other_name="other value"

        Args:
            content: String containing key-value pairs

        Returns:
            Dictionary of parsed key-value pairs
        """
        pairs = self.__split_into_pairs(content)

        result = {}
        for pair in pairs:
            key, value = self.__parse_single_pair(pair)
            if key:  # Only add if we successfully extracted a key
                result[key] = value

        return result

    def __split_into_pairs(self, content: str) -> list[str]:
        """
        Split content into key-value pairs while respecting quoted strings.

        Args:
            content: String containing key-value pairs

        Returns:
            List of individual pair strings
        """
        pairs = []
        current_pair = ""
        in_quotes = False
        quote_char = None

        for char in content:
            if self.__is_quote_start(char, in_quotes):
                in_quotes = True
                quote_char = char
            elif self.__is_quote_end(char, in_quotes, quote_char):
                in_quotes = False
                quote_char = None
            elif self.__is_pair_separator(char, in_quotes):
                pairs.append(current_pair.strip())
                current_pair = ""
                continue

            current_pair += char

        if current_pair.strip():
            pairs.append(current_pair.strip())

        return pairs

    def __is_quote_start(self, char: str, in_quotes: bool) -> bool:
        """
        Check if character starts a quoted string.

        Args:
            char: Character to check
            in_quotes: Whether currently inside quotes

        Returns:
            True if character starts a quote
        """
        return char in ['"', "'"] and not in_quotes

    def __is_quote_end(
        self, char: str, in_quotes: bool, quote_char: str | None
    ) -> bool:
        """
        Check if character ends a quoted string.

        Args:
            char: Character to check
            in_quotes: Whether currently inside quotes
            quote_char: The quote character that started the string

        Returns:
            True if character ends the current quote
        """
        return char == quote_char and in_quotes

    def __is_pair_separator(self, char: str, in_quotes: bool) -> bool:
        """
        Check if character is a pair separator (comma outside quotes).

        Args:
            char: Character to check
            in_quotes: Whether currently inside quotes

        Returns:
            True if character separates pairs
        """
        return char == "," and not in_quotes

    def __parse_single_pair(self, pair: str) -> tuple[str, Any]:
        """
        Parse a single key-value pair string.

        Args:
            pair: Single pair string (e.g., 'key="value"')

        Returns:
            Tuple of (key, parsed_value)
        """
        if "=" not in pair:
            return "", None

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        parsed_value = self.__process_pair_value(value)

        return key, parsed_value

    def __process_pair_value(self, value: str) -> Any:
        """
        Process and convert a pair value to appropriate type.

        Args:
            value: Raw value string

        Returns:
            Converted value (string, int, float, or bool)
        """
        if self.__is_quoted_string(value):
            return self.__remove_quotes(value)

        return self.__convert_unquoted_value(value)

    def __is_quoted_string(self, value: str) -> bool:
        """
        Check if value is a quoted string.

        Args:
            value: Value to check

        Returns:
            True if value is quoted
        """
        return (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        )

    def __remove_quotes(self, value: str) -> str:
        """
        Remove surrounding quotes from a value.

        Args:
            value: Quoted value

        Returns:
            Value without quotes
        """
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        return value

    def __convert_unquoted_value(self, value: str) -> Any:
        """
        Convert unquoted value to appropriate Python type.

        Args:
            value: Unquoted value string

        Returns:
            Converted value (bool, int, float, or string)
        """
        try:
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            elif value.isdigit():
                return int(value)
            elif "." in value and value.replace(".", "").isdigit():
                return float(value)
        except ValueError:
            pass  # Keep as string

        return value
