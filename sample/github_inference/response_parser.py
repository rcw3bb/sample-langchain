"""
Response parser for handling API responses and tool calls.

Author: Ron Webb
Since: 1.0.0
"""

from langchain.schema.messages import AIMessage
from .tool_call_extractor import ToolCallExtractor


class ResponseParser:
    """
    Handles parsing of API responses and extraction of tool calls.

    Author: Ron Webb
    Since: 1.0.0
    """

    def __init__(self):
        """Initialize the response parser."""
        self.__tool_extractor = ToolCallExtractor()

    def parse_response_with_tools(self, content: str) -> AIMessage:
        """
        Parse response content and extract tool calls if present.

        Args:
            content: Raw response content from the API

        Returns:
            AIMessage with tool calls extracted if present
        """
        if "Action:" in content and "Action Input:" in content:
            tool_calls = self.__tool_extractor.extract_tool_calls(content)
            if tool_calls:
                # Create AIMessage with tool calls
                message = AIMessage(content=content, tool_calls=tool_calls)
                return message

        return AIMessage(content=content)
