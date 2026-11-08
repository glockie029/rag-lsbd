from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#1.加载本地下载好的本地模型
model = SentenceTransformer(
    str(ROOT / "models" / "bge-small-zh-v1.5"),
    device="cpu",
    local_files_only=True,
    trust_remote_code=False,
)

#2.读取实验资料
# 从 UTF-8 文件中读取资料
source_path = ROOT / "notes.txt"
content = source_path.read_text(encoding="utf-8")

# 按行拆分，去掉行首尾空白，并忽略空行
texts = [
    line.strip()
    for line in content.splitlines()
    if line.strip()
]

# 空文件暂不触发数据库清空
if not texts:
    raise ValueError("notes.txt 中没有可以入库的资料")

print("从文件读取的资料：", texts)

#3 将资料转换为向量
vectors = model.encode(
    texts,
    normalize_embeddings=True,
)

print("向量数组的形状:",vectors.shape)
print("第一条向量的前五个数:",vectors[0][:5])
if len(vectors) > 1:
    print("第二条向量的前五个数:",vectors[1][:5])

#4.将问题转换为向量
question = "星河项目的服务端口是多少?"

query_vector = model.encode(
    "为这个句子生成表示以用于检索相关文章：" + question,
    normalize_embeddings=True,
)

#5.计算问题与每条资料的相似度
scores = vectors @ query_vector

best_index = int(scores.argmax())

print("-" * 50)
print("问题向量的形状：", query_vector.shape)
print("每条资料的相似度：", scores)
print("最相关的资料：", texts[best_index])

#6.打开本地数据库，目录不存在时会自动创建
client = chromadb.PersistentClient(
    path=str(ROOT / "data" / "learning"),
    settings=Settings(anonymized_telemetry=False),
)

#7.打开或创建一个集合
collection = client.get_or_create_collection(
    name="learning_notes",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)

#8.保存资料及其向量
# 按资料数量自动生成 ID
ids = [
    f"note-{i + 1:03d}"
    for i in range(len(texts))
]

# 为每条资料生成对应的来源信息
metadatas = [
    {
        "source": source_path.name,
        "chunk_index": i,
    }
    for i in range(len(texts))
]

# 查询这个文件之前保存过的记录
previous = collection.get(
    where={"source": source_path.name},
    include=["metadatas"],
)

old_ids = set(previous["ids"])
new_ids = set(ids)
# 写入数据库
collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=vectors.tolist(),
    metadatas=metadatas,
)

print("本次入库的 ID：", ids)

# 写入成功后，清理文件中已经不存在的记录
stale_ids = sorted(old_ids - new_ids)

if stale_ids:
    collection.delete(ids=stale_ids)

print("本次清理的旧 ID：", stale_ids)

#9.固定按ID读取第一条记录
record = collection.get(
    ids=["note-001"],
    include=["documents","metadatas","embeddings"],
)

print("-" * 50)
print("知识库记录数：", collection.count())
print("读回的原文：", record["documents"][0])
print("读回的来源信息：", record["metadatas"][0])
print("读回的向量维度：", len(record["embeddings"][0]))
