"""
Tool call extractor for parsing tool calls from response content.

Author: Ron Webb
Since: 1.0.0
"""

from typing import Any
from .tool_input_parser import ToolInputParser


class ToolCallExtractor:
    """
    Handles extraction of tool calls from response content.

    Author: Ron Webb
    Since: 1.0.0
    """

    def __init__(self):
        """Initialize the tool call extractor."""
        self.__input_parser = ToolInputParser()

    def extract_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """
        Extract tool calls from response content with enhanced parsing capabilities.

        Supports multiple formats:
        - Action/Action Input pairs
        - Multi-line action inputs
        - Non-JSON parameters (fallback to string)
        - Multiple tool calls in sequence

        Args:
            content: Response content containing action/input patterns

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("Action:"):
                action_name, input_content, i = self.__extract_single_tool_call(
                    lines, i
                )

                if action_name and input_content:
                    tool_call = self.__create_tool_call(
                        action_name, input_content, len(tool_calls)
                    )
                    tool_calls.append(tool_call)

                continue

            i += 1

        return tool_calls

    def __extract_single_tool_call(
        self, lines: list[str], start_index: int
    ) -> tuple[str, str, int]:
        """
        Extract a single tool call from the lines starting at the given index.

        Args:
            lines: List of content lines
            start_index: Index of the Action: line

        Returns:
            Tuple of (action_name, input_content, new_index)
        """
        line = lines[start_index].strip()
        action_name = line.replace("Action:", "").strip()

        input_content, new_index = self.__extract_action_input(lines, start_index + 1)

        return action_name, input_content, new_index

    def __extract_action_input(
        self, lines: list[str], start_index: int
    ) -> tuple[str, int]:
        """
        Extract action input content starting from the given index.

        Args:
            lines: List of content lines
            start_index: Index to start searching for Action Input

        Returns:
            Tuple of (input_content, new_index)
        """
        input_content = ""
        i = start_index

        while i < len(lines):
            current_line = lines[i].strip()
            if current_line.startswith("Action Input:"):
                input_content = current_line.replace("Action Input:", "").strip()
                i += 1

                input_content, i = self.__extract_multiline_input(
                    lines, i, input_content
                )
                break
            i += 1

        return input_content, i

    def __extract_multiline_input(
        self, lines: list[str], start_index: int, initial_content: str
    ) -> tuple[str, int]:
        """
        Extract multi-line input content until reaching a stop condition.

        Args:
            lines: List of content lines
            start_index: Index to start reading from
            initial_content: Initial content already extracted

        Returns:
            Tuple of (complete_input_content, new_index)
        """
        input_content = initial_content
        i = start_index

        while i < len(lines):
            next_line = lines[i].strip()

            if self.__is_stop_line(next_line):
                break

            if input_content and not input_content.endswith("\n"):
                input_content += "\n"
            input_content += next_line
            i += 1

        return input_content, i

    def __is_stop_line(self, line: str) -> bool:
        """
        Check if a line indicates the end of action input content.

        Args:
            line: Line to check

        Returns:
            True if this line should stop input extraction
        """
        return (
            line.startswith("Action:")
            or line.startswith("Observation:")
            or line.startswith("Final Answer:")
            or not line
        )

    def __create_tool_call(
        self, action_name: str, input_content: str, call_index: int
    ) -> dict[str, Any]:
        """
        Create a tool call dictionary from extracted action and input.

        Args:
            action_name: Name of the action/tool
            input_content: Input content for the tool
            call_index: Index for generating unique call ID

        Returns:
            Tool call dictionary
        """
        tool_input = self.__input_parser.parse_tool_input(input_content)

        return {
            "name": action_name,
            "args": tool_input,
            "id": f"call_{call_index}",
        }
