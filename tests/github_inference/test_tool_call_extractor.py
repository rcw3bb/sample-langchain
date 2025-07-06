"""
Tests for ToolCallExtractor module.

Author: Ron Webb
Since: 1.0.0
"""

from sample.github_inference.tool_call_extractor import ToolCallExtractor


class TestToolCallExtractor:
    """Test cases for ToolCallExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ToolCallExtractor()

    def test_extract_single_tool_call(self):
        """Test extracting a single tool call."""
        content = """Action: search
Action Input: {"query": "test"}
Observation: Found results"""
        
        result = self.extractor.extract_tool_calls(content)
        
        expected = [
            {
                "name": "search",
                "args": {"query": "test"},
                "id": "call_0"
            }
        ]
        assert result == expected

    def test_extract_multiple_tool_calls(self):
        """Test extracting multiple tool calls."""
        content = """Action: search
Action Input: {"query": "first"}
Observation: First results

Action: calculate
Action Input: {"expression": "2+2"}
Observation: 4"""
        
        result = self.extractor.extract_tool_calls(content)
        
        expected = [
            {
                "name": "search",
                "args": {"query": "first"},
                "id": "call_0"
            },
            {
                "name": "calculate", 
                "args": {"expression": "2+2"},
                "id": "call_1"
            }
        ]
        assert result == expected

    def test_extract_with_key_value_input(self):
        """Test extracting tool call with key-value input format."""
        content = """Action: search
Action Input: query="test search", limit=5
Observation: Results found"""
        
        result = self.extractor.extract_tool_calls(content)
        
        expected = [
            {
                "name": "search",
                "args": {"query": "test search", "limit": 5},
                "id": "call_0"
            }
        ]
        assert result == expected

    def test_extract_with_simple_string_input(self):
        """Test extracting tool call with simple string input."""
        content = """Action: echo
Action Input: hello world
Observation: hello world"""
        
        result = self.extractor.extract_tool_calls(content)
        
        expected = [
            {
                "name": "echo",
                "args": {"input": "hello world"},
                "id": "call_0"
            }
        ]
        assert result == expected

    def test_extract_multiline_input(self):
        """Test extracting tool call with multiline input."""
        content = """Action: format_text
Action Input: {
    "text": "Hello world",
    "style": "bold"
}
Observation: **Hello world**"""
        
        result = self.extractor.extract_tool_calls(content)
        
        expected = [
            {
                "name": "format_text",
                "args": {"text": "Hello world", "style": "bold"},
                "id": "call_0"
            }
        ]
        assert result == expected

    def test_no_tool_calls(self):
        """Test content with no tool calls."""
        content = "This is just regular text without any actions."
        
        result = self.extractor.extract_tool_calls(content)
        assert result == []

    def test_incomplete_tool_call(self):
        """Test content with incomplete tool call (missing Action Input)."""
        content = """Action: search
Some other text without Action Input"""
        
        result = self.extractor.extract_tool_calls(content)
        assert result == []

    def test_tool_call_with_final_answer(self):
        """Test tool call that stops at Final Answer."""
        content = """Action: search
Action Input: {"query": "test"}
Final Answer: Here is the result"""
        
        result = self.extractor.extract_tool_calls(content)
        
        expected = [
            {
                "name": "search",
                "args": {"query": "test"},
                "id": "call_0"
            }
        ]
        assert result == expected

    def test_empty_action_input(self):
        """Test tool call with empty action input."""
        content = """Action: ping
Action Input: 
Observation: pong"""
        
        result = self.extractor.extract_tool_calls(content)
        assert result == []

    def test_mixed_content_with_tools(self):
        """Test content with mixed text and tool calls."""
        content = """I need to search for information.

Action: search
Action Input: {"query": "Python programming"}

Now let me calculate something.

Action: calculate  
Action Input: {"expression": "10 * 5"}

That's all I need."""
        
        result = self.extractor.extract_tool_calls(content)
        
        expected = [
            {
                "name": "search",
                "args": {"query": "Python programming"},
                "id": "call_0"
            },
            {
                "name": "calculate",
                "args": {"expression": "10 * 5"},
                "id": "call_1"
            }
        ]
        assert result == expected
