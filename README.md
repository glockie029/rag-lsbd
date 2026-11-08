# rag-lsbd：本地 RAG 学习实验

从零编写的中文 RAG 实践项目，用于学习文档入库、向量检索、知识更新，以及调用本地语言模型生成带引用的回答。代码保留逐步学习过程，通过独立脚本观察每个环节。

采用 **BGE-small 中文向量模型 + Chroma 本地数据库 + Huihui-Qwen3.5-4B 本地生成模型**。后续学习方向包括提示词注入、知识库投毒及防御效果对照；目前已完成基础链路，安全实验尚未实施。

## 项目进度

截至 **2026-11-08**，根据当前源码和学习过程中的运行结果整理：

- [x] 独立虚拟环境、CPU 依赖安装与版本锁定。
- [x] BGE 固定提交下载、文件哈希校验和离线加载。
- [x] 中文向量化、手动余弦相似度检索。
- [x] Chroma 持久化、按 ID 读取及跨进程读回。
- [x] TXT 按非空行入库，自动生成 ID 和来源信息。
- [x] 文件资料新增、修改及非空文件内减少记录的同步。
- [x] 单条、多条资料检索，显示距离并拼接上下文。
- [x] 调用本地 Huihui，根据资料回答并标注引用。
- [x] 有答案、组合问题和资料不足问题的初步验证。
- [ ] 距离阈值过滤和程序层面的资料不足拦截。
- [ ] 更多正负样本、阈值校准和引用正确性评测。
- [ ] 多文件导入、稳定文档标识、按 token 切片。
- [ ] 配置集中管理、函数拆分和自动化回归测试。
- [ ] 提示词注入、RAG 投毒及干净/投毒知识库对照实验。
- [ ] 简易网页界面。

模型对预算问题曾返回“无法确认”；这是一次提示词约束下的观察结果，当前代码尚未实现距离阈值拒答。

## 处理流程

~~~text
notes.txt → 按非空行拆分 → BGE 向量化 → Chroma 保存
问题 → BGE 向量化 → Chroma 检索 → 拼接参考资料 → Huihui 回答及引用
~~~

Python 负责检索和发送上下文。向量模型与生成模型各自独立，不需要针对示例文档训练模型。

## 技术与环境

| 组件 | 配置 |
| --- | --- |
| 平台 | 已在 Windows 11 验证 |
| Python | 3.13 x64，安装脚本要求 PowerShell 7.4+ |
| 嵌入模型 | BAAI/bge-small-zh-v1.5，CPU，512 维归一化向量 |
| 核心依赖 | PyTorch 2.14.1+cpu、Sentence Transformers 6.2.0、Chroma 1.5.9、httpx 0.28.1 |
| 生成模型 | Huihui-Qwen3.5-4B-abliterated；原本机使用 Q4_K_M / llama.cpp Vulkan |
| 生成接口 | http://127.0.0.1:8081/v1/chat/completions |
| API 模型别名 | huihui-qwen3.5-4b-abliterated-local |

原本机使用 i5-12400F、16GB 内存、RX 6650 XT 8GB。BGE 在 CPU 上运行，生成模型通过独立服务运行。其他平台尚未验证；锁定依赖以 Windows / Python 3.13 为目标。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| Install.ps1 | 创建环境、安装依赖、下载 BGE 并自检 |
| download_model.py | 固定版本下载和文件校验 |
| embedding.lock.json | 嵌入模型提交、文件哈希及预期配置 |
| verify_environment.py | 离线嵌入、检索和跨进程持久化验证 |
| learn_rag.py | TXT 入库、相似度演示、更新与删除同步 |
| search_notes.py | 根据输入问题检索一条资料 |
| test_llm.py | 单独测试本地生成接口 |
| ask_notes.py | 检索两条资料，显示距离，生成带引用的回答 |
| notes.txt | 虚构项目的示例资料 |
| Stop-Model.ps1 | 原本机相邻模型部署目录的停止入口 |
| requirements*.txt / requirements.in | 依赖锁定与顶层依赖说明 |

## 快速开始

在 PowerShell 7 中执行。代码根据脚本目录定位模型、资料和数据库，可以克隆到自己的目录。

### 1. 克隆与安装

~~~powershell
git clone https://github.com/glockie029/rag-lsbd.git
cd rag-lsbd
./Install.ps1
~~~

下载需要代理时，可指定当前有效的 HTTP 代理，例如：

~~~powershell
./Install.ps1 -Proxy 'http://127.0.0.1:7890'
~~~

脚本通过 py -3.13 寻找解释器，也支持 -Python 参数指定 Python 3.13 x64 解释器。CPU PyTorch 从官方 CPU 源安装，其他包从 PyPI 安装。无需激活环境，后续直接使用项目解释器。

BGE 固定提交为 **7999e1d3359715c523056ef9478215996d62a620**。自检结果保存到本地 evidence/environment-verification.json，自检库与学习库分开保存。

### 2. 导入示例资料

~~~powershell
& ./.venv/Scripts/python.exe ./learn_rag.py
~~~

示例文件每条事实占一行，目前包含负责人和服务端口。修改后重新运行入库脚本，观察 ID、清理记录和读回结果。

### 3. 运行检索

~~~powershell
& ./.venv/Scripts/python.exe ./search_notes.py
~~~

输入“星河项目的默认服务端口是多少？”，查看原文、来源和余弦距离。距离越小越接近，但不是答案正确概率。

### 4. 接入生成模型

本仓库的安装脚本只准备嵌入和检索环境，**不下载或启动 Huihui / llama.cpp 服务**。

请先自行启动支持 Qwen3.5 的 llama.cpp 和本地 GGUF 模型，配置为上表中的接口地址和模型别名。已有兼容接口也可使用，需要在两个调用脚本中修改地址和模型名。生成服务准备好后执行：

~~~powershell
& ./.venv/Scripts/python.exe ./test_llm.py
& ./.venv/Scripts/python.exe ./ask_notes.py
~~~

可以测试：

- 星河项目的端口是多少？
- 星河项目由谁负责，默认服务端口是多少？
- 星河项目的预算是多少？

对照打印的参考资料检查回答和引用。资料编号按本次检索排序生成，不等同于数据库记录 ID。

Stop-Model.ps1 依赖仓库父目录下的 local-model-uncensored/Model.cmd。独立克隆后如果没有该外部部署目录，应使用自己的生成服务停止方式。

## 当前限制

- 按行读取 TXT，尚未做长文切片和上下文 token 预算。
- ID 根据片段位置生成，目前面向单个 notes.txt；多文件导入需要重新设计标识，避免冲突。
- 空文件会拒绝入库，不会自动清空数据库；旧元数据字段可能保留。
- Top-k 只返回相近候选，不能保证资料包含答案；当前尚无程序阈值过滤。
- 引用由模型生成，尚未逐项核对事实与引用。系统提示词约束不构成安全隔离。
- 目前是学习脚本，异常处理、批量评测和生产服务能力仍待补充。

## 公开范围与复现

仓库只包含源码、依赖锁定、模型来源锁定和虚构示例资料。模型权重、虚拟环境、Chroma 数据、缓存、IDE 配置与本机日志不提交，安装和运行后会在本地生成。

学习过程中已验证向量化、检索、跨进程读取、文件更新/删除同步，以及基于资料的本地模型回答。不同硬件和环境仍需自行验证。第三方模型和软件遵循其各自许可证。

## 参考

- [BGE 中文模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [Huihui 模型说明](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-abliterated)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Chroma](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
