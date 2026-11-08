from chat_api import complete_chat

# 准备发送给模型的数据
payload = {
    "messages": [
        {
            "role": "user",
            "content": "请只回复：模型接口连接成功",
        }
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": False,
}

# 调用与 RAG 问答相同的标准接口
answer = complete_chat(payload)
print("模型回答：", answer)