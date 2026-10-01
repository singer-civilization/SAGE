import json
from urllib import request

from agent.expander.config import ExpanderSettings


def call_deepseek(settings: ExpanderSettings, system_prompt: str, user_prompt: str) -> tuple[dict, str]:

    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=f"{settings.base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(http_request, timeout=settings.request_timeout_seconds) as response:
        raw_response = response.read().decode("utf-8")

    response_json = json.loads(raw_response)
    raw_text = response_json["choices"][0]["message"]["content"]
    parsed_payload = json.loads(raw_text)
    if "candidates" not in parsed_payload or not isinstance(parsed_payload["candidates"], list):
        raise RuntimeError("DeepSeek response JSON must contain a candidates list")
    return parsed_payload, raw_text
