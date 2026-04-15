# Added by Jingbin Lin, 2024-06-19
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import openai

from dojo.core.solvers.llm_helpers.backends.open_ai import FunctionSpec

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger = logging.getLogger("Backend")


class OpenRouterClient:
    PromptType = Union[str, Dict[str, Any], List[Any]]
    FunctionCallType = Dict[str, Any]
    OutputType = Union[str, FunctionCallType]

    def __init__(self, client_cfg):
        """
        Initialize an OpenRouter client using the OpenAI-compatible SDK.
        """
        self.model = client_cfg.model_id
        self.base_url = client_cfg.base_url or DEFAULT_OPENROUTER_BASE_URL
        self.api_key = (
            os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("PRIMARY_KEY_OPENROUTER", "")
            or os.getenv("PRIMARY_KEY", "")
        )

        default_headers = {}
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "")
        x_title = os.getenv("OPENROUTER_X_TITLE", "")
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if x_title:
            default_headers["X-OpenRouter-Title"] = x_title

        client_kwargs = {
            "max_retries": 3,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self._client = openai.OpenAI(**client_kwargs)

        logging.getLogger("httpx").setLevel(logging.WARNING)

    @property
    def client_content_key(self):
        return "content"

    def _query_client(
        self,
        messages: List[Dict[str, str]],
        model_kwargs: Dict[str, Any] = {},
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
    ) -> Tuple[OutputType, Dict[str, Any]]:
        model_kwargs["model"] = self.model
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        func_spec = None
        if json_schema and function_name and function_description:
            func_spec = FunctionSpec(function_name, json.loads(json_schema), function_description)

        if func_spec is not None:
            filtered_kwargs["functions"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["function_call"] = {"name": function_name}

        completion = None
        start_time = time.monotonic()
        try:
            completion = self._client.chat.completions.create(messages=messages, **filtered_kwargs)
        except openai.BadRequestError as e:
            if "function calling" in str(e).lower() or "functions" in str(e).lower():
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                filtered_kwargs.pop("functions", None)
                filtered_kwargs.pop("function_call", None)
                completion = self._client.chat.completions.create(messages=messages, **filtered_kwargs)
            else:
                raise

        latency = time.monotonic() - start_time

        choice = completion.choices[0]
        usage_stats = completion.to_dict().get("usage", {}) if completion is not None else {}
        usage_stats["latency"] = latency

        if func_spec is None or "functions" not in filtered_kwargs:
            output = choice.message.content
        else:
            function_call = choice.message.function_call
            if not function_call:
                logger.warning(
                    "No function call was used despite function spec. Fallback to text.\n"
                    f"Message content: {choice.message.content}"
                )
                output = choice.message.content
            elif not str(function_call.name).strip() == str(func_spec.name).strip():
                logger.warning(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {function_call.name}. Fallback to text."
                )
                output = choice.message.content
            else:
                try:
                    output = json.loads(function_call.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(f"Error decoding function arguments:\n{function_call.arguments}")
                    raise ex

        return output, usage_stats

    def query(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[str] = None,
        function_name: Optional[str] = None,
        function_description: Optional[str] = None,
        **model_kwargs,
    ) -> OutputType:
        """
        General LLM query for OpenRouter's OpenAI-compatible chat completions API.
        """
        output, usage_stats = self._query_client(
            messages=messages,
            model_kwargs=model_kwargs,
            json_schema=json_schema,
            function_name=function_name,
            function_description=function_description,
        )

        return output, usage_stats
