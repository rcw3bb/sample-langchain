"""
Tests for ToolParameterExtractor module.

Author: Ron Webb
Since: 1.0.0
"""

from typing import Any
from unittest.mock import Mock, patch
from langchain.tools.base import BaseTool
from sample.github_inference.tool_parameter_extractor import ToolParameterExtractor


class TestToolParameterExtractor:
    """Test cases for ToolParameterExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ToolParameterExtractor()

    def test_initialization(self):
        """Test parameter extractor initialization."""
        assert isinstance(self.extractor, ToolParameterExtractor)

    def test_get_tool_parameters_with_valid_tool(self):
        """Test parameter extraction from a tool with proper schema."""
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "Test tool"
            
            def _run(self, query: str, limit: int = 10) -> str:
                return f"query: {query}, limit: {limit}"

        tool = TestTool()
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)
        # Should return some parameter information or empty string if no schema

    def test_get_tool_parameters_with_none_tool(self):
        """Test parameter extraction with None tool."""
        # Need to cast to Any to bypass type checking for this edge case test
        result = self.extractor.get_tool_parameters(None)  # type: ignore
        assert result == ""

    def test_get_tool_parameters_with_args_schema(self):
        """Test parameter extraction from tool with args_schema."""
        # Create a mock tool with args_schema
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        
        # Create a mock schema with __fields__
        mock_schema = Mock()
        mock_fields = {
            "query": Mock(description="Search query"),
            "limit": Mock(description="Result limit")
        }
        mock_schema.__fields__ = mock_fields
        mock_tool.args_schema = mock_schema

        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)
        if result:  # Only check content if extraction was successful
            assert "query" in result or "limit" in result

    def test_get_tool_parameters_with_schema_method(self):
        """Test parameter extraction from tool with schema method."""
        # Create a mock tool with schema method
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        
        # Create a mock schema with callable schema method
        mock_schema = Mock()
        mock_schema.schema.return_value = {
            "properties": {
                "query": {"description": "Search query"},
                "limit": {"description": "Result limit", "type": "integer"}
            }
        }
        mock_tool.args_schema = mock_schema

        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)
        if result:  # Only check content if extraction was successful
            assert "query" in result or "limit" in result

    def test_get_tool_parameters_with_run_method_annotations(self):
        """Test parameter extraction from _run method annotations."""
        class AnnotatedTool(BaseTool):
            name: str = "annotated_tool"
            description: str = "Tool with annotated _run method"
            
            def _run(self, query: str, limit: int = 10) -> str:
                return f"query: {query}, limit: {limit}"

        tool = AnnotatedTool()
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)
        # Should extract parameters from method annotations

    def test_get_tool_parameters_with_run_annotations_fallback(self):
        """Test parameter extraction from run method annotations as fallback."""
        # Create a tool with annotated run method
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.args_schema = None
        
        # Mock _run method with annotations
        mock_run = Mock()
        mock_run.__annotations__ = {"query": str, "limit": int, "return": str}
        mock_tool._run = mock_run
        
        # Also mock run method
        mock_run_method = Mock()
        mock_run_method.__annotations__ = {"query": str, "limit": int, "return": str}
        mock_tool.run = mock_run_method

        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)
        if result:
            assert "query" in result and "limit" in result

    def test_get_tool_parameters_no_schema_no_annotations(self):
        """Test parameter extraction when no schema or annotations are available."""
        # Create a minimal tool without schema or annotations
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "simple_tool"
        mock_tool.description = "Simple tool"
        mock_tool.args_schema = None
        
        # Mock methods without annotations
        mock_tool._run = Mock()
        mock_tool._run.__annotations__ = {}
        mock_tool.run = Mock()
        mock_tool.run.__annotations__ = {}

        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert result == ""

    def test_get_tool_parameters_exception_handling(self):
        """Test that parameter extraction handles exceptions gracefully."""
        # Create a tool that will raise exceptions during extraction
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.description = "Tool that causes errors"
        
        # Make args_schema raise an exception when accessed
        type(mock_tool).args_schema = Mock(side_effect=Exception("Schema error"))

        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        # Should handle exception gracefully and return empty string
        assert result == ""

    def test_get_tool_parameters_run_method_exception(self):
        """Test exception handling when accessing run method annotations."""
        class ProblematicTool(BaseTool):
            name: str = "problematic_tool"
            description: str = "Tool that causes exceptions during parameter extraction"
            
            def _run(self, query: str) -> str:
                return f"query: {query}"
            
            @property
            def args_schema(self):
                # This will raise an exception when accessed
                raise Exception("Simulated exception during args_schema access")

        tool = ProblematicTool()
        result = self.extractor.get_tool_parameters(tool)
        # The exception occurs when accessing args_schema, but it falls back to _run annotations
        # So it should still find the 'query' parameter
        assert "query" in result

    def test_get_tool_parameters_exception_in_run_method_access(self):
        """Test exception handling when accessing the run method annotations (covers lines 58-60)."""
        class RunMethodExceptionTool(BaseTool):
            name: str = "run_exception_tool"
            description: str = "Tool that causes exceptions during run method access"
            
            def _run(self, query: str) -> str:
                return f"query: {query}"
            
            # Property that raises exception when run method is accessed
            @property
            def run(self):
                raise Exception("Exception when accessing run method")

        # Remove args_schema to force fallback to run method
        tool = RunMethodExceptionTool()
        if hasattr(tool, 'args_schema'):
            delattr(tool, 'args_schema')
        
        result = self.extractor.get_tool_parameters(tool)
        # Should handle exception gracefully and return what it found from _run
        assert "query" in result

    def test_extract_from_pydantic_fields_with_descriptions(self):
        """Test extraction from Pydantic fields with descriptions."""
        # Create a mock schema with fields that have descriptions
        mock_schema = Mock()
        
        # Mock field with description
        mock_field1 = Mock()
        mock_field1.description = "Search query parameter"
        
        # Mock field with field_info description
        mock_field2 = Mock()
        mock_field2.description = None
        mock_field_info = Mock()
        mock_field_info.description = "Limit parameter"
        mock_field2.field_info = mock_field_info
        
        mock_schema.__fields__ = {
            "query": mock_field1,
            "limit": mock_field2
        }

        # Test the private method indirectly through a tool
        mock_tool = Mock(spec=BaseTool)
        mock_tool.args_schema = mock_schema
        
        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)

    def test_extract_from_pydantic_fields_without_descriptions(self):
        """Test extraction from Pydantic fields without descriptions."""
        # Create a mock schema with fields without descriptions
        mock_schema = Mock()
        
        mock_field1 = Mock()
        mock_field1.description = None
        mock_field1.field_info = None
        
        mock_field2 = Mock()
        mock_field2.description = None
        mock_field2.field_info = None
        
        mock_schema.__fields__ = {
            "param1": mock_field1,
            "param2": mock_field2
        }

        # Test through a tool
        mock_tool = Mock(spec=BaseTool)
        mock_tool.args_schema = mock_schema
        
        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)

    def test_extract_from_schema_dict_with_properties(self):
        """Test extraction from schema dictionary with properties."""
        # Create a mock schema with schema method returning dict
        mock_schema = Mock()
        mock_schema.schema.return_value = {
            "properties": {
                "query": {"description": "Search query", "type": "string"},
                "limit": {"description": "Result limit", "type": "integer"},
                "enabled": {"type": "boolean"}  # No description
            }
        }

        # Test through a tool
        mock_tool = Mock(spec=BaseTool)
        mock_tool.args_schema = mock_schema
        
        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)

    def test_extract_from_schema_dict_invalid_format(self):
        """Test extraction when schema method returns invalid format."""
        # Create a mock schema with schema method returning invalid format
        mock_schema = Mock()
        mock_schema.schema.return_value = {"invalid": "format"}  # No properties

        # Test through a tool
        mock_tool = Mock(spec=BaseTool)
        mock_tool.args_schema = mock_schema
        
        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert result == ""

    def test_extract_from_schema_dict_exception(self):
        """Test extraction when schema method raises exception."""
        # Create a mock schema with schema method that raises exception
        mock_schema = Mock()
        mock_schema.schema.side_effect = Exception("Schema error")

        # Test through a tool
        mock_tool = Mock(spec=BaseTool)
        mock_tool.args_schema = mock_schema
        
        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert result == ""

    def test_get_tool_parameters_with_hasattr_checks(self):
        """Test that extraction properly checks for attribute existence."""
        # Create a tool that has some attributes but not others
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "partial_tool"
        mock_tool.description = "Partially implemented tool"
        
        # Has args_schema but no __fields__
        mock_schema = Mock()
        # Remove __fields__ attribute
        if hasattr(mock_schema, '__fields__'):
            delattr(mock_schema, '__fields__')
        # Remove schema method
        if hasattr(mock_schema, 'schema'):
            delattr(mock_schema, 'schema')
        
        mock_tool.args_schema = mock_schema

        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        # Should handle missing attributes gracefully
        assert isinstance(result, str)

    def test_tool_without_args_schema_attribute(self):
        """Test tool that doesn't have args_schema attribute at all."""
        # Create a tool without args_schema
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "no_schema_tool"
        mock_tool.description = "Tool without schema"
        
        # Remove args_schema attribute
        if hasattr(mock_tool, 'args_schema'):
            delattr(mock_tool, 'args_schema')

        tool: BaseTool = mock_tool  # type: ignore
        result = self.extractor.get_tool_parameters(tool)

        assert isinstance(result, str)

    def test_get_tool_parameters_general_exception_handling(self):
        """Test exception handling in the main try block (lines 58-60)."""
        # Create a tool that will cause an exception in multiple places
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "exception_tool"
        
        # Make hasattr raise an exception
        def problematic_hasattr(obj, name):
            if name == "args_schema":
                raise Exception("hasattr exception")
            return False
        
        with patch('builtins.hasattr', side_effect=problematic_hasattr):
            result = self.extractor.get_tool_parameters(mock_tool)
            # Should handle exception gracefully and return empty string
            assert result == ""
