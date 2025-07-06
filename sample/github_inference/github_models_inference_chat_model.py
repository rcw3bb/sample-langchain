"""
GitHubModelsInferenceChatModel module for GitHub API interactions.

Author: Ron Webb
Since: 1.0.0
"""

import asyncio
from typing import Any, Optional
import requests
from pydantic import Field
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult, ChatGeneration
from langchain.schema.messages import BaseMessage
from langchain.tools.base import BaseTool
from .rate_limiter import RateLimiter
from .message_converter import MessageConverter
from .response_parser import ResponseParser
from .system_prompt_builder import SystemPromptBuilder
from .configuration_handler import ConfigurationHandler


class GitHubModelsInferenceChatModel(BaseChatModel):
    """
    Chat model implementation for GitHub Models Inference API that works with ReAct agents.

    Author: Ron Webb
    Since: 1.0.0
    """

    model: str = Field(default="openai/gpt-4o", description="Model name to use")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Maximum tokens to generate"
    )
    timeout: Optional[int] = Field(
        default=30, ge=1, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries")

    api_key: str = Field(
        ..., exclude=True, description="GitHub token for authentication"
    )
    base_url: str = Field(
        default="https://models.github.ai/inference/chat/completions",
        description="API base URL",
    )

    headers: dict[str, str] = Field(default_factory=dict, exclude=True)

    def __init__(self, **data):
        """
        Initialize the chat model.

        Args:
            model: Model name (default: "openai/gpt-4o")
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: None)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retries (default: 3)
            api_key: GitHub token for authentication
            base_url: API base URL (default: GitHub Models API)
        """
        super().__init__(**data)
        self.__rate_limiter = RateLimiter()
        self._bound_tools: Optional[list[BaseTool]] = None
        self.__message_converter = MessageConverter()
        self.__response_parser = ResponseParser()
        self.__prompt_builder = SystemPromptBuilder()

    def model_post_init(self, __context) -> None:
        """
        Post-initialization to set headers for API requests.
        """
        headers = ConfigurationHandler.build_headers(self.api_key)
        object.__setattr__(self, "headers", headers)

    def __make_rate_limited_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: int = 30,
    ) -> requests.Response:
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
        return self.__rate_limiter.make_request(
            url, headers, payload, timeout or self.timeout or 30
        )

    @property
    def _llm_type(self) -> str:
        """
        Returns the type of the LLM.
        """
        return "github_models_inference_chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat completions from messages with tool calling support.
        """
        api_messages = self.__prepare_api_messages(messages)
        response = self.__send_chat_completion_request(api_messages)

        return self.__process_chat_response(response)

    def __prepare_api_messages(
        self, messages: list[BaseMessage]
    ) -> list[dict[str, str]]:
        """
        Convert LangChain messages to API format and enhance with tool information.

        Args:
            messages: List of LangChain messages

        Returns:
            List of API-formatted messages
        """
        api_messages = self.__message_converter.convert_messages_to_api_format(messages)

        if self._bound_tools:
            api_messages = self.__enhance_messages_with_tools(api_messages)

        return api_messages

    def __enhance_messages_with_tools(
        self, api_messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Add or enhance system message with tool descriptions.

        Args:
            api_messages: List of API-formatted messages

        Returns:
            Enhanced messages with tool information
        """
        if not api_messages or api_messages[0]["role"] != "system":
            # No system message exists, create one with tool descriptions
            system_prompt = self.__prompt_builder.build_system_prompt(self._bound_tools)
            api_messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            # Enhance existing system message with tool descriptions
            tool_descriptions = self.__prompt_builder.build_tool_descriptions(
                self._bound_tools
            )
            api_messages[0]["content"] += tool_descriptions

        return api_messages

    def __send_chat_completion_request(
        self, api_messages: list[dict[str, str]]
    ) -> requests.Response:
        """
        Send chat completion request to the API.

        Args:
            api_messages: List of API-formatted messages

        Returns:
            HTTP response from the API
        """
        payload = ConfigurationHandler.build_request_payload(
            model=self.model,
            api_messages=api_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return self.__make_rate_limited_request(
            url=self.base_url,
            headers=self.headers,
            payload=payload,
            timeout=self.timeout or 30,
        )

    def __process_chat_response(self, response: requests.Response) -> ChatResult:
        """
        Process the API response and return a ChatResult.

        Args:
            response: HTTP response from the API

        Returns:
            ChatResult containing the generated message
        """
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        message = self.__response_parser.parse_response_with_tools(content)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async generation using threading to make sync calls async.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._generate, messages, stop, run_manager, **kwargs
            )
            result = await asyncio.wrap_future(future)
            return result

    def bind_tools(self, tools: list[BaseTool]) -> "GitHubModelsInferenceChatModel":
        """
        Bind tools to the model. Store tools for use in generation.

        Args:
            tools: List of tools to bind to the model

        Returns:
            New instance of the model with tools bound
        """
        new_instance = GitHubModelsInferenceChatModel(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        new_instance._bound_tools = tools
        return new_instance
