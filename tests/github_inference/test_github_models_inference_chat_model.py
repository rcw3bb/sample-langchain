"""
Tests for the refactored GitHubModelsInferenceChatModel.

Author: Ron Webb
Since: 1.0.0
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
# from typing import List  # Using built-in list type instead
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from sample.github_inference.github_models_inference_chat_model import GitHubModelsInferenceChatModel


class TestGitHubModelsInferenceChatModel:
    """Test cases for the refactored GitHubModelsInferenceChatModel class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.model = GitHubModelsInferenceChatModel(api_key=self.api_key)

    def test_initialization(self):
        """Test model initialization."""
        assert self.model.api_key == self.api_key
        assert self.model.model == "openai/gpt-4o"
        assert self.model.temperature == 0.7
        assert self.model._bound_tools is None

    def test_llm_type(self):
        """Test _llm_type property."""
        assert self.model._llm_type == "github_models_inference_chat"

    def test_headers_initialization(self):
        """Test that headers are properly initialized."""
        expected_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        assert self.model.headers == expected_headers

    @patch('sample.github_inference.github_models_inference_chat_model.GitHubModelsInferenceChatModel._GitHubModelsInferenceChatModel__make_rate_limited_request')
    def test_generate_without_tools(self, mock_request):
        """Test generation without tools."""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_request.return_value = mock_response

        messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        result = self.model._generate(messages)

        assert result.generations[0].message.content == "Test response"
        mock_request.assert_called_once()

    def test_bind_tools(self):
        """Test tool binding functionality."""
        # Create a concrete tool class for testing
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "Test tool description"
            
            def _run(self, *args, **kwargs):
                return "test result"
        
        mock_tool = TestTool()
        tools: list[BaseTool] = [mock_tool]
        bound_model = self.model.bind_tools(tools)
        
        # Should return a new instance
        assert bound_model is not self.model
        assert bound_model._bound_tools == tools
        assert bound_model.api_key == self.api_key
        assert bound_model.model == self.model.model

    @patch('sample.github_inference.github_models_inference_chat_model.GitHubModelsInferenceChatModel._GitHubModelsInferenceChatModel__make_rate_limited_request')
    def test_generate_with_tools(self, mock_request):
        """Test generation with tools bound."""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Action: test_tool\nAction Input: {\"param\": \"value\"}"}}]
        }
        mock_request.return_value = mock_response

        # Create concrete test tool
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "Test tool description"
            
            def _run(self, *args, **kwargs):
                return "test result"
        
        # Bind tools and generate
        bound_model = self.model.bind_tools([TestTool()])
        messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        result = bound_model._generate(messages)

        # Should have tool calls in the response
        assert "Action:" in result.generations[0].message.content
        mock_request.assert_called_once()
        
        # Verify system prompt was added for tools
        call_args = mock_request.call_args
        payload = call_args[1]['payload']
        assert payload['messages'][0]['role'] == 'system'
        assert 'tools' in payload['messages'][0]['content'].lower()

    def test_model_configuration_validation(self):
        """Test model configuration with various parameters."""
        model = GitHubModelsInferenceChatModel(
            api_key="test",
            model="custom-model",
            temperature=0.5,
            max_tokens=1000,
            timeout=60,
            max_retries=5
        )
        
        assert model.model == "custom-model"
        assert model.temperature == 0.5
        assert model.max_tokens == 1000
        assert model.timeout == 60
        assert model.max_retries == 5

    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        # Check that all components are initialized
        assert hasattr(self.model, '_GitHubModelsInferenceChatModel__message_converter')
        assert hasattr(self.model, '_GitHubModelsInferenceChatModel__response_parser')
        assert hasattr(self.model, '_GitHubModelsInferenceChatModel__prompt_builder')
        assert hasattr(self.model, '_GitHubModelsInferenceChatModel__rate_limiter')

    @patch('sample.github_inference.github_models_inference_chat_model.GitHubModelsInferenceChatModel._GitHubModelsInferenceChatModel__make_rate_limited_request')
    def test_timeout_fallback_in_make_rate_limited_request(self, mock_request):
        """Test timeout fallback logic in rate limited request."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_request.return_value = mock_response

        # Create model without timeout
        model = GitHubModelsInferenceChatModel(api_key="test")
        model.timeout = None  # Explicitly set to None
        
        messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        model._generate(messages)

        # Should use default timeout of 30 when both model.timeout and passed timeout are None
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]['timeout'] == 30

    @patch('sample.github_inference.github_models_inference_chat_model.GitHubModelsInferenceChatModel._GitHubModelsInferenceChatModel__make_rate_limited_request')
    def test_enhance_existing_system_message_with_tool_descriptions(self, mock_request):
        """Test enhancing existing system message with tool descriptions."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_request.return_value = mock_response

        # Create tool
        class TestTool(BaseTool):
            name: str = "test_tool"
            description: str = "Test tool description"
            
            def _run(self, *args, **kwargs):
                return "test result"
        
        # Bind tools and create messages with existing system message
        bound_model = self.model.bind_tools([TestTool()])
        messages: list[BaseMessage] = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello")
        ]
        
        bound_model._generate(messages)

        # Check that the existing system message was enhanced
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        payload = call_args[1]['payload']
        system_message = payload['messages'][0]
        
        assert system_message['role'] == 'system'
        assert 'You are a helpful assistant.' in system_message['content']
        # Should have tool descriptions appended
        assert 'test_tool' in system_message['content']

    @patch('sample.github_inference.github_models_inference_chat_model.GitHubModelsInferenceChatModel._generate')
    @patch('asyncio.wrap_future')
    @patch('concurrent.futures.ThreadPoolExecutor')
    @pytest.mark.asyncio
    async def test_async_generation(self, mock_executor_class, mock_wrap_future, mock_generate):
        """Test async generation using ThreadPoolExecutor."""
        # Setup mocks
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        mock_future = Mock()
        mock_executor.submit.return_value = mock_future
        
        expected_result = Mock()
        
        # Create an awaitable mock for wrap_future
        async def mock_awaitable():
            return expected_result
        
        mock_wrap_future.return_value = mock_awaitable()
        
        # Test async generation
        messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        result = await self.model._agenerate(messages)

        # Verify the execution flow
        mock_executor.submit.assert_called_once_with(
            self.model._generate, messages, None, None
        )
        mock_wrap_future.assert_called_once_with(mock_future)
        assert result == expected_result

    @patch('sample.github_inference.github_models_inference_chat_model.GitHubModelsInferenceChatModel._GitHubModelsInferenceChatModel__make_rate_limited_request')
    @pytest.mark.asyncio
    async def test_async_generation_integration(self, mock_request):
        """Test async generation integration with real async call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Async test response"}}]
        }
        mock_request.return_value = mock_response

        messages: list[BaseMessage] = [HumanMessage(content="Hello async")]
        result = await self.model._agenerate(messages)
        
        assert result.generations[0].message.content == "Async test response"
        mock_request.assert_called_once()

    def test_make_rate_limited_request_timeout_logic(self):
        """Test the timeout logic in __make_rate_limited_request method through _generate."""
        # Test case where both timeout parameter and model timeout are None
        with patch.object(self.model, f'_{self.model.__class__.__name__}__rate_limiter') as mock_rate_limiter:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            mock_rate_limiter.make_request.return_value = mock_response
            
            # Set model timeout to None to test the default 30 value
            self.model.timeout = None
            
            # Call _generate which internally calls __make_rate_limited_request
            messages: list[BaseMessage] = [HumanMessage(content="Test")]
            self.model._generate(messages)
            
            # Verify that the rate limiter was called with default timeout of 30
            mock_rate_limiter.make_request.assert_called_once()
            call_args = mock_rate_limiter.make_request.call_args
            # Check positional arguments (timeout should be the 4th argument)
            assert call_args[0][3] == 30  # Check the timeout argument
