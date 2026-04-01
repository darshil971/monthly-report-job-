"""
OpenAI / Azure OpenAI utility functions for the monthly report pipeline.
Extracted from admin-backend/openai_utils.py — only the functions needed:
  - GPT4Input dataclass
  - gpt4_1_azure_call
  - gpt4o_call
  - get_embedding_model
"""

import dataclasses
import json
import os
from typing import List, Optional
import httpx
from langchain_openai import AzureOpenAIEmbeddings

from src.config import MonthlyReportJobConfig

_config = MonthlyReportJobConfig()
_http_client = httpx.Client(http1=True, timeout=30)


@dataclasses.dataclass
class GPT4Input:
    actor: str
    text: str = None
    image_url: str = None

    def is_human(self):
        return self.actor == 'user'

    def is_ai(self):
        return self.actor == 'assistant'

    def is_system(self):
        return self.actor == 'system'

    def gpt4_openai_input(self):
        return {
            'role': self.actor,
            'content': self.text
        }

    def message_chain_input(self):
        if self.image_url:
            main_body = {
                "type": "image_url",
                "image_url": {"url": self.image_url}
            }
        else:
            main_body = {
                "type": "text",
                "text": self.text
            }
        return {
            "role": self.actor,
            "content": [main_body]
        }


def gpt4_1_azure_call(
    gpt4_inputs: List[GPT4Input],
    temperature: float = 0.8,
    max_tokens: int = 800,
    timeout: int = 60
) -> Optional[str]:
    """Call Azure GPT-4.1 deployment."""
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": _config.azure_gpt41_api_key,
        }
        payload = {
            "messages": [x.message_chain_input() for x in gpt4_inputs],
            "temperature": temperature,
            "top_p": 0,
            "max_tokens": max_tokens,
        }

        response = _http_client.post(
            _config.azure_gpt41_endpoint,
            headers=headers,
            json=payload,
            timeout=timeout
        )

        if response.status_code != 200:
            print(f"API call failed with status code: {response.status_code}")
            print(f"Response text: {response.text}")
            return None

        json_resp = response.json()
        if "choices" in json_resp and len(json_resp["choices"]) > 0:
            return json_resp["choices"][0]["message"]["content"]
        else:
            print(f"No choices in response: {json_resp}")
            return None

    except Exception as e:
        print(f"Error in gpt4_1_azure_call: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def gpt4o_call(
    text,
    prompt,
    temperature=0,
    timeout=1200,
    json_needed=False
) -> Optional[str]:
    """Call Azure GPT-4o deployment."""
    headers = {
        "Content-Type": "application/json",
        "api-key": _config.azure_gpt4o_api_key,
    }
    messages = [
        {'role': "system", 'content': prompt},
        {"role": "user", "content": json.dumps(text)}
    ]
    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": 0,
        "max_tokens": 4000,
    }
    if json_needed:
        payload["response_format"] = {"type": "json_object"}

    response = _http_client.post(
        _config.azure_gpt4o_endpoint,
        headers=headers,
        json=payload,
        timeout=timeout
    )
    json_resp = response.json()

    if "choices" in json_resp and len(json_resp["choices"]) > 0:
        message_content = json_resp["choices"][0]["message"]["content"]
        return message_content.replace("```json", "").replace("```", "").strip()
    return None


def get_embedding_model() -> AzureOpenAIEmbeddings:
    """Return Azure OpenAI embedding model (text-embedding-3-large, 1024D)."""
    print(f"using {_config.azure_embedding_deployment} for embeddings")
    return AzureOpenAIEmbeddings(
        deployment=_config.azure_embedding_deployment,
        azure_endpoint=_config.azure_embedding_endpoint,
        openai_api_key=_config.azure_embedding_api_key,
        openai_api_type="azure",
        openai_api_version="2024-02-01",
        dimensions=1024,
        timeout=10,
    )


# ============================================================
# LLMResponse — lightweight wrapper used by gpt_5_2_chat / gpt_5_nano
# ============================================================

@dataclasses.dataclass
class LLMResponse:
    content: str
    token_data: Optional[dict] = None
    additional_args: Optional[dict] = None

    def to_json_str(self) -> str:
        return json.dumps({"content": self.content})


# ============================================================
# GPT-5.2-chat (Azure Sweden Central) — used by theme clustering
# ============================================================

# Azure GPT-5.2-chat
AZURE_GPT52_API_KEY = os.getenv(
    "AZURE_GPT52_API_KEY",
    "f339ca0a17e943df9f2ed92be64dadcb"
)
AZURE_GPT52_ENDPOINT = os.getenv(
    "AZURE_GPT52_ENDPOINT",
    "https://openai-sweden-central-deployment.openai.azure.com/openai/deployments/gpt-5.2-chat/chat/completions?api-version=2024-12-01-preview"
)

# Azure GPT-5-nano
AZURE_GPT5NANO_API_KEY = os.getenv(
    "AZURE_GPT5NANO_API_KEY",
    "8268d8ed232440be940c3f0e4f8b05d3"
)
AZURE_GPT5NANO_ENDPOINT = os.getenv(
    "AZURE_GPT5NANO_ENDPOINT",
    "https://openai-sweden-central-deployment.openai.azure.com/openai/deployments/gpt-5-nano/chat/completions?api-version=2025-04-01-preview"
)


def gpt_5_2_chat(
    input_list: List[GPT4Input],
    temperature: float = 0.8,
    max_tokens: int = 800,
    timeout: int = 15,
    streaming: bool = False,
    response_consumer=None,
    request_context=None,
) -> LLMResponse:
    """Call Azure GPT-5.2-chat deployment (non-streaming only for batch jobs)."""
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_GPT52_API_KEY,
    }
    payload = {
        "messages": [inp.message_chain_input() for inp in input_list],
    }
    try:
        response = _http_client.post(
            AZURE_GPT52_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        json_resp = response.json()
        content = json_resp['choices'][0]['message']['content']
        return LLMResponse(content=content, token_data={})
    except Exception as e:
        print(f"[gpt_5_2_chat] Error: {e}, falling back to gpt4_1_azure_call")
        # Fallback to GPT-4.1
        result = gpt4_1_azure_call(input_list, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        return LLMResponse(content=result or "", token_data={})


def gpt_5_nano(
    input_list: List[GPT4Input],
    temperature: float = 0.8,
    max_tokens: int = 800,
    timeout: int = 15,
    streaming: bool = False,
    response_consumer=None,
    request_context=None,
) -> LLMResponse:
    """Call Azure GPT-5-nano deployment."""
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_GPT5NANO_API_KEY,
    }
    payload = {
        "messages": [inp.message_chain_input() for inp in input_list],
    }
    try:
        response = _http_client.post(
            AZURE_GPT5NANO_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        json_resp = response.json()
        content = json_resp['choices'][0]['message']['content']
        return LLMResponse(content=content, token_data={})
    except Exception as e:
        print(f"[gpt_5_nano] Error: {e}, falling back to gpt4_1_azure_call")
        result = gpt4_1_azure_call(input_list, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        return LLMResponse(content=result or "", token_data={})
