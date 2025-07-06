"""
System prompt builder for generating tool-aware system prompts.

Author: Ron Webb
Since: 1.0.0
"""

from typing import Optional
from langchain.tools.base import BaseTool
from .tool_parameter_extractor import ToolParameterExtractor


class SystemPromptBuilder:
    """
    Handles building of system prompts with tool descriptions.

    Author: Ron Webb
    Since: 1.0.0
    """

    BASE_PROMPT = """You are a helpful assistant that must use the provided tools to solve problems.

IMPORTANT: You must use the available tools for any task that requires computation, 
data retrieval, or external operations. Follow this exact pattern:

1. Analyze the user's request and identify which tools are needed
2. Use tools in this format:
   Action: tool_name
   Action Input: {"parameter": "value"}
3. Wait for the Observation: containing the tool result
4. If you need more information or want to use another tool, repeat steps 2-3
5. Once you have all needed information, provide your final answer

CRITICAL RULES:
- Never perform calculations manually - always use calculator tools
- Never guess or make up information - use search/retrieval tools
- Always wait for Observation: before proceeding
- You can use multiple tools in sequence to solve complex problems
- Format parameters correctly based on tool requirements 
  (JSON, key-value pairs, or simple strings)"""

    def __init__(self):
        """Initialize the system prompt builder."""
        self.__parameter_extractor = ToolParameterExtractor()

    def build_tool_descriptions(self, tools: Optional[list[BaseTool]]) -> str:
        """
        Generate comprehensive tool descriptions for the system prompt.

        Includes tool names, descriptions, and parameter information when available.

        Args:
            tools: List of tools to generate descriptions for

        Returns:
            Formatted tool descriptions string with usage examples
        """
        if not tools:
            return ""

        descriptions = []
        for tool in tools:
            tool_desc = ""

            # Get basic tool info
            if hasattr(tool, "name") and hasattr(tool, "description"):
                tool_desc = f"- {tool.name}: {tool.description}"

                # Try to get parameter information
                param_info = self.__parameter_extractor.get_tool_parameters(tool)
                if param_info:
                    tool_desc += f"\n  Parameters: {param_info}"

                descriptions.append(tool_desc)

        if descriptions:
            tool_list = "\n".join(descriptions)
            return f"""

Available tools:
{tool_list}

Tool Usage Instructions:
- Format: Action: tool_name
- Next line: Action Input: {{"param1": value1, "param2": value2}}
- Alternative formats supported:
  * Simple string: Action Input: search query
  * Key-value: Action Input: query="search term", limit=5
  * Multi-line JSON is supported
- Wait for "Observation:" with the tool result before proceeding
- You can use multiple tools in sequence to solve complex problems"""

        return ""

    def build_system_prompt(self, tools: Optional[list[BaseTool]]) -> str:
        """
        Build complete system prompt with tool descriptions.

        Args:
            tools: List of tools to include in the prompt

        Returns:
            Complete system prompt string
        """
        return self.BASE_PROMPT + self.build_tool_descriptions(tools)
