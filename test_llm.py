import httpx

# 本地模型的聊天接口
url = "http://127.0.0.1:8081/v1/chat/completions"

# 准备发送给模型的数据
payload = {
    "model": "huihui-qwen3.5-4b-abliterated-local",
    "messages": [
        {
            "role": "user",
            "content": "请只回复：本地模型连接成功",
        }
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}

# 发送请求，并检查 HTTP 状态
with httpx.Client(trust_env=False, timeout=180.0) as client:
    response = client.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

# 从响应中提取模型回答
answer = data["choices"][0]["message"]["content"]
print("模型回答：", answer)