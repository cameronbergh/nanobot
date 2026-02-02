"""LiteLLM provider implementation for multi-provider support."""

import os
from typing import Any

import asyncio
import litellm
from litellm import acompletion
from rlm.core.rlm import RLM

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, and many other providers through
    a unified interface.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5"
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        
        # Detect OpenRouter by api_key prefix or explicit api_base
        self.is_openrouter = (
            (api_key and api_key.startswith("sk-or-")) or
            (api_base and "openrouter" in api_base)
        )
        
        # Track if using custom endpoint (vLLM, etc.)
        self.is_vllm = bool(api_base) and not self.is_openrouter
        
        # Configure LiteLLM based on provider
        if api_key:
            if self.is_openrouter:
                # OpenRouter mode - set key
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif self.is_vllm:
                # vLLM/custom endpoint - uses OpenAI-compatible API
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic" in default_model:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            elif "openai" in default_model or "gpt" in default_model:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
        
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = model or self.default_model
        
        # For OpenRouter, prefix model name if not already prefixed
        if self.is_openrouter and not model.startswith("openrouter/"):
            model = f"openrouter/{model}"
        
        # For vLLM, use hosted_vllm/ prefix per LiteLLM docs
        # Convert openai/ prefix to hosted_vllm/ if user specified it
        if self.is_vllm:
            model = f"hosted_vllm/{model}"
        
        # Configure backend kwargs for RLM
        backend_kwargs: dict[str, Any] = {
            "model_name": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if self.api_key:
            backend_kwargs["api_key"] = self.api_key
        if self.api_base:
            backend_kwargs["api_base"] = self.api_base

        # Instantiate RLM
        rlm = RLM(
            backend="litellm",
            backend_kwargs=backend_kwargs,
            verbose=False
        )
        
        # Extract root prompt from last user message if available
        root_prompt = None
        if messages and messages[-1].get("role") == "user":
            root_prompt = str(messages[-1].get("content", ""))

        try:
            # Run RLM completion in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: rlm.completion(prompt=messages, root_prompt=root_prompt)
            )
            return self._parse_rlm_response(result, model)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling RLM: {str(e)}",
                finish_reason="error",
            )
    
    def _parse_rlm_response(self, result: Any, model: str) -> LLMResponse:
        """Parse RLM response into our standard format."""

        # Calculate usage
        usage = {}
        if result.usage_summary and result.usage_summary.model_usage_summaries:
            # Try to get usage for the requested model, or aggregate
            if model in result.usage_summary.model_usage_summaries:
                 summary = result.usage_summary.model_usage_summaries[model]
                 usage = {
                     "prompt_tokens": summary.total_input_tokens,
                     "completion_tokens": summary.total_output_tokens,
                     "total_tokens": summary.total_input_tokens + summary.total_output_tokens
                 }
            else:
                # Aggregate if specific model not found or multiple models
                total_input = 0
                total_output = 0
                for summary in result.usage_summary.model_usage_summaries.values():
                    total_input += summary.total_input_tokens
                    total_output += summary.total_output_tokens
                usage = {
                     "prompt_tokens": total_input,
                     "completion_tokens": total_output,
                     "total_tokens": total_input + total_output
                 }

        return LLMResponse(
            content=result.response,
            tool_calls=[], # RLM handles tools internally
            finish_reason="stop",
            usage=usage,
        )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
