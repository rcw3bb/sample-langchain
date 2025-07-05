"""
GitHubModelsInferenceLLM module with ReAct agent implementation.

Author: Ron Webb
Since: 1.0.0
"""

import os
import time
from typing import ClassVar, Dict, List, Any, Optional
from dotenv import load_dotenv
from pydantic import Field
import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult, ChatGeneration
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from .commons import RateLimiter


class GitHubModelsInferenceChatModel(BaseChatModel):
    """
    Chat model implementation for GitHub Models Inference API that works with ReAct agents.
    """
    token: str = Field(..., exclude=True)
    model_id: str
    api_url: ClassVar[str] = "https://models.github.ai/inference/chat/completions"
    headers: Dict[str, str] = Field(default_factory=dict, exclude=True)

    def __init__(self, **data):
        """
        Initialize the chat model.
        """
        super().__init__(**data)
        self.__rate_limiter = RateLimiter()
        self._bound_tools: Optional[List[Any]] = None

    def model_post_init(self, __context) -> None:
        """
        Post-initialization to set headers for API requests.
        """
        object.__setattr__(
            self,
            "headers",
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def __make_rate_limited_request(self, url: str, headers: Dict[str, str], 
                                  payload: Dict[str, Any], timeout: int = 30) -> requests.Response:
        """
        Make a rate-limited HTTP request using the internal rate limiter.
        
        Args:
            url: The URL to make the request to
            headers: HTTP headers for the request
            payload: JSON payload for the request
            timeout: Request timeout in seconds
            
        Returns:
            HTTP response object
        """
        return self.__rate_limiter.make_request(url, headers, payload, timeout)

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the LLM.
        """
        return "github_models_inference_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat completions from messages.
        """        # Convert LangChain messages to API format
        api_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                # Enhance system message with tool descriptions
                content = str(message.content) if message.content else ""
                enhanced_content = content + self._get_tool_descriptions()
                api_messages.append({"role": "system", "content": enhanced_content})
            elif isinstance(message, HumanMessage):
                api_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                api_messages.append({"role": "assistant", "content": message.content})
        
        # If no system message exists, add one with tool descriptions
        if not api_messages or api_messages[0]["role"] != "system":
            system_prompt = """You are a helpful assistant that must use the provided tools to solve problems. Follow this pattern:

1. Think about what tools you need to use
2. Use Action: tool_name format followed by Action Input: with the parameters
3. Wait for the tool result (marked as Observation:)
4. Continue until you have the final answer

Never calculate directly - always use the available tools.""" + self._get_tool_descriptions()
            api_messages.insert(0, {"role": "system", "content": system_prompt})
        
        #print(f"API Messages: {api_messages}")# Prepare the payload - keep it simple for GitHub Models API
        payload = {
            "model": self.model_id, 
            "messages": api_messages, 
            "stream": False
        }
        
        # Use rate limiter to make the request
        response = self.__make_rate_limited_request(
            url=self.api_url,
            headers=self.headers,
            payload=payload,
            timeout=30
        )
        
        data = response.json()
        #print(f"API Response: {data}")
        
        content = data["choices"][0]["message"]["content"]
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async generation is not supported.
        """
        raise NotImplementedError("Async generation not supported")

    def bind_tools(self, tools: Any) -> "GitHubModelsInferenceChatModel":
        """
        Bind tools to the model. Store tools for use in generation.
        """
        # Create a new instance with tools bound
        new_instance = GitHubModelsInferenceChatModel(
            token=self.token,
            model_id=self.model_id
        )
        new_instance._bound_tools = tools
        return new_instance

    def _get_tool_descriptions(self) -> str:
        """
        Generate tool descriptions for the system prompt.
        """
        if not hasattr(self, '_bound_tools') or not self._bound_tools:
            return ""
        
        descriptions = []
        for tool in self._bound_tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                descriptions.append(f"- {tool.name}: {tool.description}")
        
        if descriptions:
            return f"\n\nAvailable tools:\n" + "\n".join(descriptions) + "\n\nTo use a tool, format your response as: Action: tool_name\nAction Input: {{\"param1\": value1, \"param2\": value2}}"
        return ""


@tool
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        The sum of a and b
    """
    print(f"Tool: Adding {a} and {b}")
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiplies two numbers and returns the result.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        
    Returns:
        The product of a and b
    """
    print(f"Tool: Multiplying {a} and {b}")
    return a * b

TOOLS = [add_numbers, multiply_numbers]


def manual_react_loop(llm_instance, tools, question: str, max_iterations: int = 5):
    """
    Manual ReAct loop implementation for demonstration.
    """
    # Create a tool mapping
    tool_map = {tool.name: tool for tool in tools}
    
    # Create system message with tool descriptions
    tool_descriptions = []
    for tool in tools:
        if hasattr(tool, 'name') and hasattr(tool, 'description'):
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
    
    system_content = """You are a helpful assistant that must use the provided tools to solve problems. Follow this pattern:

1. Think about what tools you need to use
2. Use Action: tool_name format followed by Action Input: with the parameters
3. Wait for the tool result (marked as Observation:)
4. Continue until you have the final answer

Never calculate directly - always use the available tools.

Available tools:
""" + "\n".join(tool_descriptions) + """

To use a tool, format your response as:
Action: tool_name
Action Input: {"param1": value1, "param2": value2}"""
    
    # Start the conversation with system message
    messages: List[BaseMessage] = [
        SystemMessage(content=system_content),
        HumanMessage(content=question)
    ]
    
    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        # Get response from LLM
        result = llm_instance._generate(messages)
        ai_response = result.generations[0].message.content
        print(f"AI Response: {ai_response}")
        
        # Check if response contains an action
        if "Action:" in ai_response and "Action Input:" in ai_response:
            # Parse the action
            lines = ai_response.split('\n')
            action_line = None
            input_line = None
            
            for line in lines:
                if line.strip().startswith("Action:"):
                    action_line = line.strip().replace("Action:", "").strip()
                elif line.strip().startswith("Action Input:"):
                    input_line = line.strip().replace("Action Input:", "").strip()
            
            if action_line and input_line and action_line in tool_map:
                try:
                    # Parse the input JSON
                    import json
                    tool_input = json.loads(input_line)
                    
                    # Execute the tool
                    tool = tool_map[action_line]
                    print(f"Executing tool: {action_line} with input: {tool_input}")
                    tool_result = tool.invoke(tool_input)
                    print(f"Tool result: {tool_result}")
                    
                    # Add messages to conversation
                    messages.append(AIMessage(content=ai_response))
                    messages.append(HumanMessage(content=f"Observation: {tool_result}"))
                    
                except Exception as e:
                    print(f"Error executing tool: {e}")
                    break
            else:
                print(f"Could not parse action or tool not found. Action: {action_line}")
                break
        else:
            # No action found, assume final answer
            print("Final answer received")
            messages.append(AIMessage(content=ai_response))
            return ai_response
    
    return "Max iterations reached"


if __name__ == "__main__":
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    MODEL_ID = "openai/gpt-4o"

    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN is not set in the .env file.")

    print("Initializing chat model...")
    llm_instance = GitHubModelsInferenceChatModel(token=GITHUB_TOKEN, model_id=MODEL_ID)
    
    TOOL_QUESTION = "Calculate 7 + 5 then multiply the result by 2."
    print(f"Tool Question: {TOOL_QUESTION}")
    
    print("Using manual ReAct loop...")
    
    try:
        print("Starting manual ReAct agent...")
        result = manual_react_loop(llm_instance, TOOLS, TOOL_QUESTION)
        
        print(f"\n=== Final Answer ===")
        print(f"{result}")
    except Exception as exc:
        print(f"Error occurred: {exc}")
