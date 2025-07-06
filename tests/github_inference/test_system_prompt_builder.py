"""
Tests for SystemPromptBuilder module.

Author: Ron Webb
Since: 1.0.0
"""

from typing import Any
from unittest.mock import Mock
from langchain.tools.base import BaseTool
from sample.github_inference.system_prompt_builder import SystemPromptBuilder


class TestSystemPromptBuilder:
    """Test cases for SystemPromptBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.prompt_builder = SystemPromptBuilder()

    def test_initialization(self):
        """Test prompt builder initialization."""
        assert isinstance(self.prompt_builder, SystemPromptBuilder)
        # Check that parameter extractor is initialized
        assert hasattr(self.prompt_builder, '_SystemPromptBuilder__parameter_extractor')

    def test_build_system_prompt_with_tools(self):
        """Test building system prompt with tools."""
        # Create a concrete tool class for testing
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "Test tool description"
            
            def _run(self, *args: Any, **kwargs: Any) -> str:
                return "test result"

        tools: list[BaseTool] = [TestTool()]
        result = self.prompt_builder.build_system_prompt(tools)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "action" in result.lower()
        assert "available tools" in result.lower()
        assert "test_tool" in result

    def test_build_system_prompt_with_none_tools(self):
        """Test building system prompt with None tools."""
        result = self.prompt_builder.build_system_prompt(None)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "action" in result.lower()
        # Should still have basic prompt even without tools

    def test_build_system_prompt_with_empty_tools(self):
        """Test building system prompt with empty tools list."""
        result = self.prompt_builder.build_system_prompt([])

        assert isinstance(result, str)
        assert len(result) > 0
        assert "action" in result.lower()

    def test_build_tool_descriptions_with_tools(self):
        """Test building tool descriptions with tools."""
        # Create concrete tool classes for testing
        class TestTool1(BaseTool):
            name: str = "tool_one"
            description: str = "First test tool"
            
            def _run(self, *args: Any, **kwargs: Any) -> str:
                return "result one"

        class TestTool2(BaseTool):
            name: str = "tool_two"
            description: str = "Second test tool"
            
            def _run(self, *args: Any, **kwargs: Any) -> str:
                return "result two"

        tools: list[BaseTool] = [TestTool1(), TestTool2()]
        result = self.prompt_builder.build_tool_descriptions(tools)

        assert isinstance(result, str)
        assert "tool_one" in result
        assert "tool_two" in result
        assert "First test tool" in result
        assert "Second test tool" in result
        assert "Available tools:" in result

    def test_build_tool_descriptions_with_none_tools(self):
        """Test building tool descriptions with None tools."""
        result = self.prompt_builder.build_tool_descriptions(None)
        assert result == ""

    def test_build_tool_descriptions_with_empty_tools(self):
        """Test building tool descriptions with empty tools list."""
        result = self.prompt_builder.build_tool_descriptions([])
        assert result == ""

    def test_build_tool_descriptions_with_tool_missing_attributes(self):
        """Test building tool descriptions with tool missing name/description attributes."""
        # Create a mock tool without proper name/description using Mock with spec
        mock_tool = Mock(spec=BaseTool)
        # Simulate missing attributes by having them raise AttributeError
        type(mock_tool).name = Mock(side_effect=AttributeError)
        type(mock_tool).description = Mock(side_effect=AttributeError)

        tools: list[BaseTool] = [mock_tool]  # type: ignore
        result = self.prompt_builder.build_tool_descriptions(tools)

        # Should handle gracefully and return empty string or basic format
        assert isinstance(result, str)

    def test_build_tool_descriptions_includes_parameters(self):
        """Test that tool descriptions include parameter information when available."""
        # Create a tool with parameter schema
        class ParameterizedTool(BaseTool):
            name: str = "param_tool"
            description: str = "Tool with parameters"
            
            def _run(self, query: str, limit: int = 10) -> str:
                return f"query: {query}, limit: {limit}"

        tools: list[BaseTool] = [ParameterizedTool()]
        result = self.prompt_builder.build_tool_descriptions(tools)

        assert "param_tool" in result
        assert "Tool with parameters" in result
        # Should include tool usage instructions
        assert "Tool Usage Instructions:" in result
        assert "Action:" in result
        assert "Action Input:" in result

    def test_build_tool_descriptions_format_includes_usage_instructions(self):
        """Test that tool descriptions include proper usage instructions."""
        class SimpleTool(BaseTool):
            name: str = "simple_tool"
            description: str = "Simple tool"
            
            def _run(self, *args: Any, **kwargs: Any) -> str:
                return "simple result"

        tools: list[BaseTool] = [SimpleTool()]
        result = self.prompt_builder.build_tool_descriptions(tools)

        # Check for key instruction elements
        assert "Action: tool_name" in result
        assert "Action Input:" in result
        assert "Observation:" in result
        assert "JSON" in result or "json" in result

    def test_build_system_prompt_includes_tool_descriptions(self):
        """Test that system prompt includes tool descriptions when tools are provided."""
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "Test tool description"
            
            def _run(self, *args: Any, **kwargs: Any) -> str:
                return "test result"

        tools: list[BaseTool] = [TestTool()]
        system_prompt = self.prompt_builder.build_system_prompt(tools)
        tool_descriptions = self.prompt_builder.build_tool_descriptions(tools)

        # System prompt should include tool descriptions
        assert "test_tool" in system_prompt
        # Should contain key elements from tool descriptions
        assert "Available tools:" in system_prompt

    def test_build_system_prompt_react_agent_format(self):
        """Test that system prompt follows ReAct agent format."""
        class TestTool(BaseTool):
            name: str = "search"
            description: str = "Search for information"
            
            def _run(self, *args: Any, **kwargs: Any) -> str:
                return "search result"

        tools: list[BaseTool] = [TestTool()]
        result = self.prompt_builder.build_system_prompt(tools)

        # Should include ReAct methodology elements
        assert "action" in result.lower()
        assert "observation" in result.lower()

    def test_tool_descriptions_alternative_input_formats(self):
        """Test that tool descriptions mention alternative input formats."""
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "Test tool"
            
            def _run(self, *args: Any, **kwargs: Any) -> str:
                return "result"

        tools: list[BaseTool] = [TestTool()]
        result = self.prompt_builder.build_tool_descriptions(tools)

        # Should mention alternative formats
        assert "Simple string" in result or "Key-value" in result
        assert "Alternative formats supported" in result or "formats supported" in result

    def test_build_tool_descriptions_handles_missing_name(self):
        """Test that build_tool_descriptions handles tools without name attribute."""
        # Create a mock tool that has description but no name
        mock_tool = Mock(spec=BaseTool)
        mock_tool.description = "Test description"
        # Make name attribute raise AttributeError
        type(mock_tool).name = Mock(side_effect=AttributeError)

        tools: list[BaseTool] = [mock_tool]  # type: ignore
        result = self.prompt_builder.build_tool_descriptions(tools)

        # Should handle gracefully
        assert isinstance(result, str)

    def test_build_tool_descriptions_handles_missing_description(self):
        """Test that build_tool_descriptions handles tools without description attribute."""
        # Create a mock tool that has name but no description
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "test_tool"
        # Make description attribute raise AttributeError
        type(mock_tool).description = Mock(side_effect=AttributeError)

        tools: list[BaseTool] = [mock_tool]  # type: ignore
        result = self.prompt_builder.build_tool_descriptions(tools)

        # Should handle gracefully
        assert isinstance(result, str)

    def test_build_tool_descriptions_empty_after_filtering(self):
        """Test build_tool_descriptions with tools that don't meet criteria."""
        # Create tools that will fail the hasattr checks (no name, no description)
        mock_tool1 = Mock(spec=BaseTool)
        # Remove name and description attributes completely
        if hasattr(mock_tool1, 'name'):
            delattr(mock_tool1, 'name')
        if hasattr(mock_tool1, 'description'):
            delattr(mock_tool1, 'description')
        
        mock_tool2 = Mock(spec=BaseTool)
        if hasattr(mock_tool2, 'name'):
            delattr(mock_tool2, 'name')
        if hasattr(mock_tool2, 'description'):
            delattr(mock_tool2, 'description')

        tools: list[BaseTool] = [mock_tool1, mock_tool2]  # type: ignore
        result = self.prompt_builder.build_tool_descriptions(tools)

        # Should handle gracefully - either empty or with mock names
        assert isinstance(result, str)

    def test_build_tool_descriptions_with_none_tools_early_return(self):
        """Test early return when tools is None (line 59)."""
        # This tests the early return path: if not tools: return ""
        result = self.prompt_builder.build_tool_descriptions(None)
        assert result == ""

    def test_build_tool_descriptions_with_empty_list_early_return(self):
        """Test early return when tools is empty list (line 59)."""
        # This tests the early return path: if not tools: return ""
        result = self.prompt_builder.build_tool_descriptions([])
        assert result == ""

    def test_build_system_prompt_comprehensive_with_tools(self):
        """Test comprehensive system prompt building with all components."""
        class ComprehensiveTool(BaseTool):
            name: str = "comprehensive_tool"
            description: str = "A comprehensive test tool with parameters"
            
            def _run(self, query: str, limit: int = 10, enabled: bool = True) -> str:
                return f"Processed: {query} with limit {limit}, enabled: {enabled}"

        tools: list[BaseTool] = [ComprehensiveTool()]
        result = self.prompt_builder.build_system_prompt(tools)

        # Should include all ReAct components
        assert "comprehensive_tool" in result
        assert "A comprehensive test tool with parameters" in result
        assert "Action:" in result
        assert "Action Input:" in result
        assert "Observation:" in result
        
        # Should include usage instructions
        assert "Tool Usage Instructions:" in result
        assert "Alternative formats supported:" in result

    def test_build_system_prompt_without_tools(self):
        """Test system prompt building without tools."""
        result = self.prompt_builder.build_system_prompt(None)
        
        # Should still have basic ReAct prompt structure
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention tool usage methodology
        assert "action" in result.lower() or "tools" in result.lower()

    def test_tool_parameter_extraction_integration(self):
        """Test integration with tool parameter extractor."""
        class ParameterTool(BaseTool):
            name: str = "param_tool"
            description: str = "Tool with parameters"
            
            def _run(self, search_query: str, max_results: int = 5) -> str:
                return f"Searching for {search_query}, max {max_results} results"

        tools: list[BaseTool] = [ParameterTool()]
        result = self.prompt_builder.build_tool_descriptions(tools)

        assert "param_tool" in result
        assert "Tool with parameters" in result
        # Should include parameter information if extractor found any
        assert isinstance(result, str)
