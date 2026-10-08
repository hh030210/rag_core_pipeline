import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8911/v1"),
    api_key=os.getenv("LLM_API_KEY", "sdu-cookie"),
    timeout=180,
)
response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL", "DeepSeek-V4-Pro"),
    temperature=0.1,
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
    messages=[{
        "role": "user",
        "content": (
            "只输出合法JSON，不要解释。给定两个记录，抽取它们的地点和时间：\n"
            "输出格式：{\"id1\": {\"地点\": [{\"label\": \"值\", \"evidence\": \"原文\", \"confidence\": 0.9}]}, \"id2\": {}}\n"
            "id1：北京故宫位于北京市东城区。每天上午八点半开放。\n"
            "id2：这是一段没有地点和时间的信息。"
        ),
    }],
)
message = response.choices[0].message
print("message_fields", list(message.model_dump().keys()))
print("content_repr", repr(message.content))
print("reasoning_repr", repr(getattr(message, "reasoning_content", None)))
