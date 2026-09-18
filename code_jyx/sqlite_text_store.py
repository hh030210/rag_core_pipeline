# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# import sqlite3
# from typing import List, Dict, Optional
# from datasets import load_dataset

# class TextStore:
#     def __init__(self, db_path: str = "texts.db", table_name: str = "EcomQueries"):
#         """
#         初始化 SQLite 文本数据库
#         """
#         self.db_path = db_path
#         self.table_name = table_name # 动态表名
#         self._init_db()

#     def _init_db(self):
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         cursor.execute(f"""
#             CREATE TABLE IF NOT EXISTS {self.table_name} (
#                 pk INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
#                 id TEXT UNIQUE,                        -- 原始数据id
#                 text TEXT NOT NULL
#             )
#         """)

#         conn.commit()
#         conn.close()

#     # ===============================
#     # 批量插入文本
#     # ===============================
#     def insert_batch(self, docs: List[Dict[str, str]]):
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         cursor.executemany(f"""
#             INSERT OR REPLACE INTO {self.table_name} (id, text)
#             VALUES (?, ?)
#         """, [(doc["id"], doc["text"]) for doc in docs])

#         conn.commit()
#         conn.close()

#     # ===============================
#     # 单条查询
#     # ===============================
#     def get_text(self, doc_id: str) -> Optional[str]:
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         cursor.execute(f"""
#             SELECT text FROM {self.table_name} WHERE id = ?
#         """, (str(doc_id),)) # 确保 ID 是字符串

#         result = cursor.fetchone()
#         conn.close()

#         return result[0] if result else None

#     # ===============================
#     # 批量查询
#     # ===============================
#     def get_texts(self, ids: List[str]) -> Dict[str, str]:
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         placeholder = ",".join(["?"] * len(ids))

#         cursor.execute(f"""
#             SELECT id, text FROM {self.table_name}
#             WHERE id IN ({placeholder})
#         """, ids)

#         rows = cursor.fetchall()
#         conn.close()

#         return {row[0]: row[1] for row in rows}

#      # ===============================
#     # 统计数据条数
#     # ===============================
#     def count(self) -> int:
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         cursor.execute(f"""
#             SELECT COUNT(*) FROM {self.table_name}
#         """)

#         result = cursor.fetchone()
#         conn.close()

#         return result[0] if result else 0    

    
#     # ===============================
#     # 删除当前表
#     # ===============================
#     def drop_table(self):
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")

#         conn.commit()
#         conn.close()

#     def get_range(self, start: int, limit: int = 10):
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         cursor.execute(f"""
#             SELECT pk, id, text FROM {self.table_name}
#             ORDER BY pk
#             LIMIT ? OFFSET ?
#         """, (limit, start))

#         rows = cursor.fetchall()
#         conn.close()

#         return [
#             {"pk": row[0], "id": row[1], "text": row[2]}
#             for row in rows
#         ]

# if __name__ == "__main__":

#     # 读取 queries
#     # bdataset = load_dataset("C-MTEB/CovidRetrieval", split="corpus")

#     store = TextStore("texts.db", "CmedqaRetrieval")

#     # # 先删掉旧表
#     # store.drop_table()

#     # # 重新建表
#     # store._init_db()

#     # # 构建 docs，使用源 id
#     # docs = []
#     # for item in dataset:
#     #     docs.append({
#     #         "id": str(item["id"]),   # ⭐ 使用原始 id
#     #         "text": item["text"]
#     #     })

#     # # # 批量插入
#     # store.insert_batch(docs)

#     # print("写入完成")

#     # 测试查询
#     # text = store.get_text('6de9047548bdcbcf1b19f32336ef2504')   
#     # print("查询结果:", text)

#     # # 统计条数
#     total = store.count()
#     print(f"当前表中共有 {total} 条数据")

#     # results = store.get_range(850, 10)

#     # for item in results:
#     #     print(item["id"], item["text"])

import os
import json
import sqlite3
from typing import List, Dict, Optional

class TextStore:
    def __init__(self, db_path: str = "lexrag.db", table_name: str = "LegalArticles"):
        """
        初始化 SQLite 文本数据库
        """
        self.db_path = db_path
        self.table_name = table_name # 动态表名
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                pk INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
                id TEXT UNIQUE,                        -- 原始数据id
                text TEXT NOT NULL                     -- 文本或JSON化的内容
            )
        """)

        conn.commit()
        conn.close()

    # ===============================
    # 批量插入文本
    # ===============================
    def insert_batch(self, docs: List[Dict[str, str]]):
        if not docs:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 对于1.7万条数据，直接批量执行效率很高
        cursor.executemany(f"""
            INSERT OR REPLACE INTO {self.table_name} (id, text)
            VALUES (?, ?)
        """, [(doc["id"], doc["text"]) for doc in docs])

        conn.commit()
        conn.close()

    # ===============================
    # 单条及批量查询
    # ===============================
    def get_text(self, doc_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT text FROM {self.table_name} WHERE id = ?
        """, (str(doc_id),)) 

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def get_texts(self, ids: List[str]) -> Dict[str, str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholder = ",".join(["?"] * len(ids))
        cursor.execute(f"""
            SELECT id, text FROM {self.table_name}
            WHERE id IN ({placeholder})
        """, ids)

        rows = cursor.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0    

    def drop_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        conn.commit()
        conn.close()

    def get_range(self, start: int, limit: int = 10):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT pk, id, text FROM {self.table_name}
            ORDER BY pk
            LIMIT ? OFFSET ?
        """, (limit, start))
        rows = cursor.fetchall()
        conn.close()
        return [{"pk": row[0], "id": row[1], "text": row[2]} for row in rows]

# ===============================
# 数据解析工具函数 (针对 LexRAG)
# ===============================
def load_lexrag_corpus(file_path: str) -> List[Dict[str, str]]:
    """读取 LexRAG 的法条语料库 (law_library.jsonl)"""
    docs =[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            
            # 兼容多种常见的语料库键名提取 id
            doc_id = str(item.get("id", item.get("doc_id", f"law_{idx}")))
            
            # 提取法条文本内容，若结构比较复杂，将其统一 dump 成 JSON 字符串保留全量信息
            if "text" in item:
                text_content = item["text"]
            elif "content" in item:
                text_content = item["content"]
            else:
                text_content = json.dumps(item, ensure_ascii=False)
                
            docs.append({
                "id": doc_id,
                "text": text_content
            })
    return docs


def load_lexrag_queries(file_path: str) -> List[Dict[str, str]]:
    """读取 LexRAG 的多轮对话查询集 (dataset.json)"""
    docs =[]
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # 应对不同包装格式：如果是字典(含 "data" 键)或直接是列表
        items = data.get("data", data) if isinstance(data, dict) else data

        for idx, item in enumerate(items):
            conv_id = str(item.get("id", item.get("conversation_id", f"conv_{idx}")))
            
            # 因为 LexRAG 是包含多轮(5轮)对话的评测集，将其完整序列化存入 text 中
            # 方便后续 RAG 模型在检索时读取对话上下文（History）并生成 prompt
            text_content = json.dumps(item, ensure_ascii=False)
            
            docs.append({
                "id": conv_id,
                "text": text_content
            })
    return docs

def get_first_turn_law_mapping(db_path: str = "lexrag_texts.db") -> Dict[str, List[str]]:
    """
    从数据库中提取：对话ID -> 第一问相关法条(Ground Truth Law IDs) 的映射
    """
    query_store = TextStore(db_path, "Conversations")
    total = query_store.count()
    
    # 获取表中所有对话数据
    records = query_store.get_range(0, total)
    
    mapping = {}
    
    for record in records:
        # 1. 获取当前对话的全局 ID (例如 "1")
        conv_id = str(record["id"]) 
        
        # 2. 将存储的字符串反序列化为 JSON 字典
        data = json.loads(record["text"]) 
        
        # 3. 提取对话列表
        conversation_list = data.get("conversation",[])
        
        if conversation_list and len(conversation_list) > 0:
            # 4. 锁定第一问（索引为 0 的元素）
            first_turn = conversation_list[0]
            
            # 5. 获取第一问的相关法条列表
            # 从你提供的 JSON 结构看，法条 ID 存放在 "article" 键中
            law_ids = first_turn.get("article", [])
            
            mapping[conv_id] = law_ids
        else:
            # 如果存在异常的空对话数据，赋空列表
            mapping[conv_id] =[]
            
    return mapping

if __name__ == "__main__":
    
    # # 假设你已经通过 git clone 存放在了当前目录的 LexRAG 文件夹下
    # base_dir = "LexRAG/data"
    # corpus_path = os.path.join(base_dir, "law_library.jsonl")
    # queries_path = os.path.join(base_dir, "dataset.json")
    
    # db_name = "lexrag_texts.db"

    # # ==========================================
    # # 1. 处理并写入 法条候选库 (Corpus)
    # # ==========================================
    # if os.path.exists(corpus_path):
    #     print(f"[{corpus_path}] 存在，开始处理法条语料...")
    #     corpus_store = TextStore(db_name, "LegalArticles")
    #     corpus_store.drop_table() # 如果重新跑，先清空旧数据
    #     corpus_store._init_db()
        
    #     corpus_docs = load_lexrag_corpus(corpus_path)
    #     corpus_store.insert_batch(corpus_docs)
        
    #     print(f"✅ 法条写入完成，表中共有 {corpus_store.count()} 条数据")
        
    #     # 简单测试抽取一条
    #     sample = corpus_store.get_range(0, 1)
    #     if sample:
    #         print(f"   -> 样例 [ID: {sample[0]['id']}]: {sample[0]['text'][:80]}...\n")
    # else:
    #     print(f"❌ 未找到法条文件：{corpus_path} (请确保已执行 git clone 并在正确目录下运行)")


    # # ==========================================
    # # 2. 处理并写入 多轮对话测试集 (Queries)
    # # ==========================================
    # if os.path.exists(queries_path):
    #     print(f"[{queries_path}] 存在，开始处理多轮对话查询集...")
    #     query_store = TextStore(db_name, "Conversations")
    #     query_store.drop_table() 
    #     query_store._init_db()
        
    #     query_docs = load_lexrag_queries(queries_path)
    #     query_store.insert_batch(query_docs)
        
    #     print(f"✅ 对话查询集写入完成，表中共有 {query_store.count()} 条数据")
        
    #     # 简单测试抽取一条
    #     sample = query_store.get_range(0, 1)
    #     if sample:
    #         print(f"   -> 样例 [ID: {sample[0]['id']}]: {sample[0]['text'][:80]}...")
    # else:
    #     print(f"❌ 未找到查询文件：{queries_path}")

    db_name = "lexrag_texts.db"
    
    print("开始提取第一问法条映射...")
    
    # 1. 调用提取方法
    gt_mapping = get_first_turn_law_mapping(db_name)
    
    # 2. 打印基础统计信息
    total_extracted = len(gt_mapping)
    print(f"✅ 成功提取了 {total_extracted} 个对话的第一问法条映射！\n")
    
    # 3. 打印前 5 个映射关系作为抽样测试
    print("====== 抽样测试 (前 5 条映射) ======")
    sample_count = 0
    for conv_id, law_ids in gt_mapping.items():
        print(f"对话 ID : {conv_id}")
        print(f"第一问对应法条 : {law_ids}")
        print("-" * 40)
        
        sample_count += 1
        if sample_count >= 5:
            break
            
    # 4. 随便抽查一个你刚刚发给我的对应 ID (比如 ID 为 "1" 的数据)
    test_id = "1"
    if test_id in gt_mapping:
        print(f"\n🔍 特别检查 ID={test_id} 的映射情况:")
        print(f"提取出的法条: {gt_mapping[test_id]}")
        # 预期输出应该包含:['《中华人民共和国民法典》第四百二十八条']
    
