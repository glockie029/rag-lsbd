# rag-lsbd：RAG 学习实验

从零编写的中文 RAG 实践项目：读取 TXT 资料，使用 BGE 向量化并存入 Chroma，检索相关原文后，通过 **OpenAI 兼容 Chat Completions 接口**调用生成模型，得到带资料编号引用的回答。

代码保留逐步学习过程。文档入库、更新/删除同步、检索、上下文拼接和提示词逻辑沿用原版本。本版本移除了 PowerShell 脚本和模型下载安装、自检文件；生成服务由使用者独立提供，不绑定特定模型或运行时。

## 项目进度

截至 **2026-11-08**：

- [x] 中文文本向量化与余弦相似度检索。
- [x] Chroma 持久化保存和按 ID 读回。
- [x] TXT 非空行入库、自动生成 ID 和来源信息。
- [x] 当前文件新增、修改及非空文件内减少记录的同步。
- [x] 单条、多条检索，打印距离并拼接参考资料。
- [x] 生成模型通过可配置的 OpenAI 兼容接口调用。
- [x] 统一接口地址、模型名和可选 Bearer 密钥配置。
- [x] 本地学习过程已验证已知事实、组合问题和资料不足时的回答。
- [ ] 距离阈值筛选和程序层面的资料不足拦截。
- [ ] 更多正负样本、阈值校准及引用正确性评测。
- [ ] 多文件导入、稳定文档标识、按 token 切片。
- [ ] 配置与业务函数进一步整理。
- [ ] 提示词注入、知识库投毒与防御对照实验。
- [ ] 简易网页界面。

模型曾对预算问题返回“无法确认”，不代表程序已有可靠的拒答机制。当前距离阈值过滤尚未实施。

## 处理流程

~~~text
notes.txt → 按非空行拆分 → BGE → Chroma
问题 → BGE → Chroma 检索 → 拼接原文和问题 → Chat Completions API → 回答及引用
~~~

BGE 向量化保留，以保持原来的检索行为。生成服务既可以是本地服务，也可以是远程 OpenAI 兼容服务。

## 文件

| 文件 | 用途 |
| --- | --- |
| learn_rag.py | TXT 入库、相似度演示、更新和删除同步 |
| search_notes.py | 根据问题检索一条资料 |
| ask_notes.py | 检索两条资料并调用生成接口回答 |
| chat_api.py | 标准聊天 HTTP 请求、配置、认证和响应解析 |
| test_llm.py | 单独检查生成服务连接 |
| notes.txt | 虚构项目示例资料 |
| .env.example | 配置变量示例，不含真实密钥 |
| requirements*.txt / requirements.in | 依赖版本和安装来源说明 |

仓库不再包含生成模型部署/停止脚本、嵌入模型下载脚本、模型哈希清单或模型环境自检脚本。

## 准备环境

依赖锁定以 Windows / Python 3.13 x64 为目标，其他平台需根据兼容性自行调整。

~~~sh
git clone https://github.com/glockie029/rag-lsbd.git
cd rag-lsbd
python -m venv .venv
~~~

激活虚拟环境后安装：

~~~sh
python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu -r requirements.cpu.lock.txt
python -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.lock.txt
python -m pip check
~~~

依赖仍包含 CPU PyTorch、Sentence Transformers、Chroma 和 httpx，用于原有 BGE 检索流程。HTTP 对话使用 httpx，不要求安装 OpenAI Python SDK。

请自行准备完整的 **BAAI/bge-small-zh-v1.5** 本地模型目录，放在项目下的 **models/bge-small-zh-v1.5/**。目录需要包含权重、分词器和 Sentence Transformers 配置。脚本继续使用本地 CPU、512 维归一化向量，不会通过聊天接口替换嵌入算法。没有准备该目录时，入库/检索脚本会加载失败。

## 配置聊天接口

脚本读取进程环境变量：

| 变量 | 含义 |
| --- | --- |
| OPENAI_BASE_URL | API 基础地址，默认 https://api.openai.com/v1；填写到 /v1，不要填写完整 /chat/completions |
| OPENAI_MODEL | 必填，填写所用服务支持的模型名称 |
| OPENAI_API_KEY | Bearer 密钥；OpenAI 官方服务必填，不需鉴权的本地兼容服务可以留空 |

PowerShell 会话示例（只是配置命令，不需要 PowerShell 脚本）：

~~~powershell
$env:OPENAI_BASE_URL = 'https://api.openai.com/v1'
$env:OPENAI_MODEL = '填写你的服务提供的模型名'
$env:OPENAI_API_KEY = '填写你的密钥'
~~~

Bash 会话示例：

~~~sh
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="your-model-name"
export OPENAI_API_KEY="your-api-key"
~~~

本地兼容服务可以把基础地址改为自己的地址，例如 http://127.0.0.1:8081/v1，并填写实际模型名。本地回环地址绕过环境代理，远程服务使用当前网络环境。示例配置不会自动加载：请在终端或 IDE 中设置变量；不要把真实密钥写进源码。

请求采用 POST /chat/completions，发送 model、messages、max_tokens、temperature 和 stream 字段，从 choices[0].message.content 读取文本。不同服务/模型对可选参数的支持可能不同，如需调整输出参数，请按服务方文档配置。已去掉专用的 chat_template_kwargs，不再绑定某个模型的思考模式。

## 运行

先检查接口，再导入资料并检索：

~~~sh
python test_llm.py
python learn_rag.py
python search_notes.py
python ask_notes.py
~~~

首次使用先运行入库脚本创建 learning_notes 集合。示例问题：

- 星河项目的默认服务端口是多少？
- 星河项目由谁负责，默认服务端口是多少？
- 星河项目的预算是多少？

对照打印的参考资料核对回答。资料编号按本次候选排序生成，不等同于数据库记录 ID。

模型、虚拟环境、数据库、缓存、IDE 配置、日志及真实环境配置均被忽略。若使用远程生成接口，检索出的片段和问题会发送到你配置的服务。

## 接口测试

~~~sh
python -m unittest discover -s tests -v
~~~

测试使用模拟 HTTP 响应，覆盖标准请求与认证、无鉴权本地服务、配置缺失、HTTP 错误和响应解析，不会请求真实模型或产生 API 费用。

## 已知限制

- 目前按行读取单个 TXT，尚未做长文切片与上下文预算。
- ID 根据位置生成，多文件场景需要重新设计标识。
- 空文件会拒绝入库，不会自动清空数据库；旧元数据字段可能保留。
- Top-k 不保证候选包含答案，当前没有程序层面的距离过滤。
- 引用由模型生成，尚无逐项事实与引用校验。提示词约束不是安全隔离机制。
- 目前是学习脚本，异常处理、批量评测和生产服务能力仍待完善。
- 接口协议测试使用模拟响应；实际服务需要自行配置并验证，不把模拟测试当作供应商可用性证明。

## 参考

- [OpenAI Chat Completions 接口](https://developers.openai.com/api/reference/resources/chat)
- [BGE 中文模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [Chroma](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
