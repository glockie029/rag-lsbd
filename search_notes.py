from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent

# 1. 加载本地向量模型，用来编码问题
model = SentenceTransformer(
    str(ROOT / "models" / "bge-small-zh-v1.5"),
    device="cpu",
    local_files_only=True,
    trust_remote_code=False,
)

# 2. 打开之前保存的数据库和集合
client = chromadb.PersistentClient(
    path=str(ROOT / "data" / "learning"),
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_collection(
    name="learning_notes",
    embedding_function=None,
)

# 3. 输入问题并生成向量
question = input("请输入问题：").strip()

query_vector = model.encode(
    "为这个句子生成表示以用于检索相关文章：" + question,
    normalize_embeddings=True,
)

# 4. 从数据库中检索最相关的一条资料
result = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=1,
    include=["documents", "metadatas", "distances"],
)

# 5. 查看检索结果
print("知识库记录数：", collection.count())
print("命中的 ID：", result["ids"][0][0])
print("命中的原文：", result["documents"][0][0])
print("余弦距离：", result["distances"][0][0])
print("来源信息：", result["metadatas"][0][0])