"""
Tests for MessageConverter module.

Author: Ron Webb
Since: 1.0.0
"""

# from typing import List  # Using built-in list type instead
from langchain.schema.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from sample.github_inference.message_converter import MessageConverter


class TestMessageConverter:
    """Test cases for MessageConverter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = MessageConverter()

    def test_convert_human_message(self):
        """Test conversion of HumanMessage."""
        messages: list[BaseMessage] = [HumanMessage(content="Hello, world!")]
        result = self.converter.convert_messages_to_api_format(messages)
        
        expected = [{"role": "user", "content": "Hello, world!"}]
        assert result == expected

    def test_convert_system_message(self):
        """Test conversion of SystemMessage."""
        messages: list[BaseMessage] = [SystemMessage(content="You are a helpful assistant.")]
        result = self.converter.convert_messages_to_api_format(messages)
        
        expected = [{"role": "system", "content": "You are a helpful assistant."}]
        assert result == expected

    def test_convert_ai_message_without_tools(self):
        """Test conversion of AIMessage without tool calls."""
        messages: list[BaseMessage] = [AIMessage(content="I can help you with that.")]
        result = self.converter.convert_messages_to_api_format(messages)
        
        expected = [{"role": "assistant", "content": "I can help you with that."}]
        assert result == expected

    def test_convert_ai_message_with_tools(self):
        """Test conversion of AIMessage with tool calls."""
        tool_calls = [
            {"name": "search", "args": {"query": "test"}, "id": "call_123"}
        ]
        messages: list[BaseMessage] = [AIMessage(content="Searching...", tool_calls=tool_calls)]
        result = self.converter.convert_messages_to_api_format(messages)
        
        expected_content = 'Searching...\n\nAction: search\nAction Input: {"query": "test"}'
        expected = [{"role": "assistant", "content": expected_content}]
        assert result == expected

    def test_convert_tool_message(self):
        """Test conversion of ToolMessage."""
        messages: list[BaseMessage] = [ToolMessage(content="Search results found", tool_call_id="call_123")]
        result = self.converter.convert_messages_to_api_format(messages)
        
        expected = [{"role": "user", "content": "Observation: Search results found"}]
        assert result == expected

    def test_convert_mixed_messages(self):
        """Test conversion of mixed message types."""
        messages: list[BaseMessage] = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            ToolMessage(content="Tool result", tool_call_id="call_123")
        ]
        result = self.converter.convert_messages_to_api_format(messages)
        
        expected = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Observation: Tool result"}
        ]
        assert result == expected

    def test_convert_empty_content(self):
        """Test conversion with empty content."""
        messages: list[BaseMessage] = [
            SystemMessage(content=""),
            HumanMessage(content=""),
            AIMessage(content="")
        ]
        result = self.converter.convert_messages_to_api_format(messages)
        
        expected = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""}
        ]
        assert result == expected
