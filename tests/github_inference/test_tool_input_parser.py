"""
Tests for ToolInputParser module.

Author: Ron Webb
Since: 1.0.0
"""

from sample.github_inference.tool_input_parser import ToolInputParser
from unittest.mock import patch


class TestToolInputParser:
    """Test cases for ToolInputParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ToolInputParser()

    def test_parse_json_input(self):
        """Test parsing valid JSON input."""
        input_content = '{"query": "test", "limit": 5}'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"query": "test", "limit": 5}
        assert result == expected

    def test_parse_key_value_pairs(self):
        """Test parsing key-value pairs."""
        input_content = 'query="test search", limit=10'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"query": "test search", "limit": 10}
        assert result == expected

    def test_parse_single_quoted_string(self):
        """Test parsing single quoted string."""
        input_content = "'search term'"
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "search term"}
        assert result == expected

    def test_parse_double_quoted_string(self):
        """Test parsing double quoted string."""
        input_content = '"another search"'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "another search"}
        assert result == expected

    def test_parse_plain_string(self):
        """Test parsing plain string without quotes."""
        input_content = "simple_search"
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "simple_search"}
        assert result == expected

    def test_parse_empty_input(self):
        """Test parsing empty input."""
        result = self.parser.parse_tool_input("")
        assert result == {}

    def test_parse_whitespace_input(self):
        """Test parsing whitespace-only input."""
        result = self.parser.parse_tool_input("   ")
        assert result == {}

    def test_parse_key_value_with_boolean(self):
        """Test parsing key-value pairs with boolean values."""
        input_content = 'enabled=true, verbose=false'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"enabled": True, "verbose": False}
        assert result == expected

    def test_parse_key_value_with_numbers(self):
        """Test parsing key-value pairs with numeric values."""
        input_content = 'count=42, ratio=3.14'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"count": 42, "ratio": 3.14}
        assert result == expected

    def test_parse_single_key_value(self):
        """Test parsing single key-value pair."""
        input_content = 'name=value'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"name": "value"}
        assert result == expected

    def test_parse_complex_quoted_values(self):
        """Test parsing complex quoted values in key-value pairs."""
        input_content = 'message="Hello, world!", count=5, flag=true'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"message": "Hello, world!", "count": 5, "flag": True}
        assert result == expected

    def test_parse_invalid_json_fallback(self):
        """Test fallback when JSON parsing fails."""
        input_content = '{invalid json'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "{invalid json"}
        assert result == expected

    def test_parse_string_with_special_characters_no_equals(self):
        """Test parsing string with brackets but no equals sign (line 67)."""
        input_content = "search{query}"
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "search{query}"}
        assert result == expected

    def test_parse_exception_handling(self):
        """Test exception handling in parse_tool_input (lines 76-78)."""
        # Create a scenario that might cause an exception during parsing
        # This is a bit tricky to test directly, but we can ensure it handles edge cases
        input_content = '{"malformed": json, "missing": quote}'
        result = self.parser.parse_tool_input(input_content)
        
        # Should fallback to string input when JSON parsing fails
        expected = {"input": '{"malformed": json, "missing": quote}'}
        assert result == expected

    def test_parse_key_value_type_conversion_error(self):
        """Test type conversion error handling in key-value parsing (line 140-141)."""
        # Test a case where type conversion might fail
        input_content = 'count=invalid_number, flag=maybe'
        result = self.parser.parse_tool_input(input_content)
        
        # Should handle conversion errors gracefully
        expected = {"count": "invalid_number", "flag": "maybe"}
        assert result == expected

    def test_parse_key_value_float_conversion(self):
        """Test float conversion in key-value parsing (line 130)."""
        input_content = 'ratio=3.14159, percentage=85.5'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"ratio": 3.14159, "percentage": 85.5}
        assert result == expected

    def test_parse_key_value_malformed_float(self):
        """Test malformed float handling in key-value parsing."""
        input_content = 'value=3.14.15'  # Invalid float with multiple dots
        result = self.parser.parse_tool_input(input_content)
        
        # Should keep as string when float conversion fails
        expected = {"value": "3.14.15"}
        assert result == expected

    def test_parse_multiline_json_with_nested_structures(self):
        """Test parsing of complex nested JSON structures."""
        input_content = '''
        {
            "query": "test search",
            "options": {
                "limit": 10,
                "filters": ["type1", "type2"]
            },
            "enabled": true
        }
        '''
        result = self.parser.parse_tool_input(input_content)
        
        expected = {
            "query": "test search",
            "options": {
                "limit": 10,
                "filters": ["type1", "type2"]
            },
            "enabled": True
        }
        assert result == expected

    def test_parse_key_value_with_quoted_comma(self):
        """Test key-value parsing with comma inside quoted values."""
        input_content = 'message="Hello, world!", count=5'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"message": "Hello, world!", "count": 5}
        assert result == expected

    def test_parse_key_value_mixed_quote_types(self):
        """Test key-value parsing with mixed quote types."""
        input_content = 'single=\'value\', double="another value", unquoted=plain'
        result = self.parser.parse_tool_input(input_content)
        
        expected = {
            "single": "value",
            "double": "another value", 
            "unquoted": "plain"
        }
        assert result == expected

    def test_parse_exception_in_main_parsing(self):
        """Test exception handling in main parsing logic."""
        parser = ToolInputParser()
        
        # Create a mock that raises an exception during parsing
        with patch.object(parser, '_ToolInputParser__parse_key_value_pairs', side_effect=Exception("Test error")):
            result = parser.parse_tool_input("test=value")
            
            # Should fallback to string input
            assert result == {"input": "test=value"}

    def test_parse_single_quoted_string_edge_case(self):
        """Test parsing of single-quoted string with no other indicators."""
        parser = ToolInputParser()
        
        # Test a single-quoted string that's the entire input
        result = parser.parse_tool_input("'single quoted string with spaces'")
        assert result == {"input": "single quoted string with spaces"}

    def test_parse_plain_string_no_special_chars(self):
        """Test parsing of plain string with no special characters or equals."""
        parser = ToolInputParser()
        
        # Test a plain string with no special characters
        result = parser.parse_tool_input("plainstring")
        assert result == {"input": "plainstring"}
        
        # Test a plain string with spaces but no special chars
        result = parser.parse_tool_input("plain string with spaces")
        assert result == {"input": "plain string with spaces"}

    def test_parse_single_quoted_string_exact_case(self):
        """Test exact case for single-quoted string parsing (line 67)."""
        parser = ToolInputParser()
        
        # Test single-quoted string that hits the specific condition:
        # - starts and ends with single quotes
        # - no curly braces, square brackets, or equals signs
        # - should return unwrapped content
        result = parser.parse_tool_input("'hello world'")
        assert result == {"input": "hello world"}
        
        # Test another case
        result = parser.parse_tool_input("'test123'")
        assert result == {"input": "test123"}

    def test_debug_single_quote_path(self):
        """Debug test to understand code path for single quotes."""
        parser = ToolInputParser()
        
        # Create an input that should definitely hit the single-quote path
        test_input = "'test'"
        
        # Manual verification of conditions:
        # 1. input_content is truthy: yes
        # 2. no {, [, = chars: yes
        # 3. does NOT start/end with double quotes: yes
        # 4. DOES start/end with single quotes: yes
        
        assert test_input.startswith("'")
        assert test_input.endswith("'")
        assert not test_input.startswith('"')
        assert not test_input.endswith('"')
        assert not any(char in test_input for char in ["{", "[", "="])
        
        result = parser.parse_tool_input(test_input)
        expected = {"input": "test"}
        assert result == expected, f"Expected {expected}, got {result}"

    def test_parse_plain_string_without_quotes(self):
        """Test parsing plain string without quotes (covers line 67)."""
        input_content = "plain_search_term"
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "plain_search_term"}
        assert result == expected

    def test_parse_string_with_special_chars_but_no_structure(self):
        """Test parsing string with no structural characters."""
        input_content = "search-term_with-special.chars"
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "search-term_with-special.chars"}
        assert result == expected

    def test_parse_plain_string_exact_else_case(self):
        """Test parsing plain string that hits the exact else case (line 67)."""
        # This should be a string that:
        # 1. Has content (not empty)
        # 2. Doesn't contain {, [, or = characters
        # 3. Doesn't start and end with quotes
        input_content = "simple_string_no_quotes"
        result = self.parser.parse_tool_input(input_content)
        
        expected = {"input": "simple_string_no_quotes"}
        assert result == expected

    def test_parse_string_exactly_line_67(self):
        """Test specific case that should hit line 67 (the else clause)."""
        # This should be a string that:
        # 1. Has content (not empty) 
        # 2. Does NOT contain {, [, or = 
        # 3. Does NOT start and end with double quotes
        # 4. Does NOT start and end with single quotes
        test_cases = [
            "word",
            "hello_world", 
            "test123",
            "file.txt",
            "some-string",
            "'incomplete_quote",
            '"incomplete_quote',
            "quote'in'middle",
            'quote"in"middle',
        ]
        
        for input_content in test_cases:
            result = self.parser.parse_tool_input(input_content)
            expected = {"input": input_content}
            assert result == expected, f"Failed for input: {input_content}"
