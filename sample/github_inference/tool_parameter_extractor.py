"""
Tool parameter extractor for handling tool parameter extraction and processing.

Author: Ron Webb
Since: 1.0.0
"""

from langchain.tools.base import BaseTool


class ToolParameterExtractor:
    """
    Handles extraction of parameter information from tools for documentation.

    Author: Ron Webb
    Since: 1.0.0
    """

    def get_tool_parameters(self, tool: BaseTool) -> str:
        """
        Extract parameter information from a tool for documentation.

        Args:
            tool: The tool to extract parameter info from

        Returns:
            String description of tool parameters
        """
        try:
            params = self.__extract_from_args_schema(tool)
            if params:
                return params

            params = self.__extract_from_method_annotations(tool)
            if params:
                return params

        except Exception:
            pass

        return ""

    def __extract_from_args_schema(self, tool: BaseTool) -> str:
        """
        Extract parameters from tool's args_schema.

        Args:
            tool: The tool to extract parameters from

        Returns:
            String description of parameters, empty if not available
        """
        if not (hasattr(tool, "args_schema") and tool.args_schema):
            return ""

        schema = tool.args_schema

        # Handle Pydantic models
        if hasattr(schema, "__fields__"):
            return self.__extract_from_pydantic_fields(schema)

        # Try to get schema dict
        if hasattr(schema, "schema") and callable(getattr(schema, "schema", None)):
            return self.__extract_from_schema_dict(schema)

        return ""

    def __extract_from_method_annotations(self, tool: BaseTool) -> str:
        """
        Extract parameters from tool method annotations as fallback.

        Args:
            tool: The tool to extract parameters from

        Returns:
            String description of parameters, empty if not available
        """
        params = self.__extract_from_run_annotations(tool, "_run")
        if params:
            return params

        params = self.__extract_from_run_annotations(tool, "run")
        if params:
            return params

        return ""

    def __extract_from_run_annotations(self, tool: BaseTool, method_name: str) -> str:
        """
        Extract parameters from a specific run method's annotations.

        Args:
            tool: The tool to extract parameters from
            method_name: Name of the method to check ("run" or "_run")

        Returns:
            Comma-separated string of parameter names, empty if not available
        """
        if not hasattr(tool, method_name):
            return ""

        method = getattr(tool, method_name)
        if not hasattr(method, "__annotations__"):
            return ""

        annotations = getattr(method, "__annotations__", {})
        params = [param for param in annotations.keys() if param != "return"]

        return ", ".join(params) if params else ""

    def __extract_from_pydantic_fields(self, schema) -> str:
        """
        Extract parameters from Pydantic model fields.

        Args:
            schema: Pydantic schema with __fields__ attribute

        Returns:
            Comma-separated string of parameter descriptions
        """
        params = []
        fields = getattr(schema, "__fields__", {})
        for field_name, field_info in fields.items():
            param_desc = field_name
            # Try different ways to get description
            if hasattr(field_info, "description") and field_info.description:
                param_desc += f" ({field_info.description})"
            elif hasattr(field_info, "field_info"):
                field_data = getattr(field_info, "field_info", None)
                if (
                    field_data
                    and hasattr(field_data, "description")
                    and field_data.description
                ):
                    param_desc += f" ({field_data.description})"
            params.append(param_desc)
        return ", ".join(params)

    def __extract_from_schema_dict(self, schema) -> str:
        """
        Extract parameters from schema dictionary.

        Args:
            schema: Schema with callable schema method

        Returns:
            Comma-separated string of parameter descriptions
        """
        try:
            schema_method = getattr(schema, "schema")
            schema_dict = schema_method()
            if isinstance(schema_dict, dict) and "properties" in schema_dict:
                params = []
                properties = schema_dict["properties"]
                if isinstance(properties, dict):
                    for prop_name, prop_info in properties.items():
                        param_desc = prop_name
                        if isinstance(prop_info, dict) and "description" in prop_info:
                            param_desc += f" ({prop_info['description']})"
                        params.append(param_desc)
                    return ", ".join(params)
        except Exception:
            pass

        return ""
