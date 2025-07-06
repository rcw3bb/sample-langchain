"""
Message converter for converting between LangChain messages and API format.

Author: Ron Webb
Since: 1.0.0
"""

import json
from typing import Any
from langchain.schema.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

# Constants for API message formatting
ROLE_KEY = "role"
CONTENT_KEY = "content"
ROLE_USER = "user"


class MessageConverter:
    """
    Handles conversion between LangChain messages and API format.

    Author: Ron Webb
    Since: 1.0.0
    """

    @staticmethod
    def convert_messages_to_api_format(
        messages: list[BaseMessage],
    ) -> list[dict[str, Any]]:
        """
        Convert LangChain messages to API format.

        Args:
            messages: List of LangChain messages

        Returns:
            List of messages in API format
        """
        api_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                content = str(message.content) if message.content else ""
                api_messages.append({ROLE_KEY: "system", CONTENT_KEY: content})
            elif isinstance(message, HumanMessage):
                api_messages.append({ROLE_KEY: ROLE_USER, CONTENT_KEY: message.content})
            elif isinstance(message, AIMessage):
                # Handle tool calls in AI messages
                content = message.content if message.content else ""
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        content += (
                            f"\n\nAction: {tool_call['name']}\n"
                            f"Action Input: {json.dumps(tool_call['args'])}"
                        )
                api_messages.append({ROLE_KEY: "assistant", CONTENT_KEY: content})
            elif isinstance(message, ToolMessage):
                # Convert tool message to user message with observation
                api_messages.append(
                    {
                        ROLE_KEY: ROLE_USER,
                        CONTENT_KEY: f"Observation: {message.content}",
                    }
                )

        return api_messages
