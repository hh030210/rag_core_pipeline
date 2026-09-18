# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# import json
# import random
# import numpy as np
# from tqdm import tqdm
# from datasets import load_dataset
# from pymilvus import MilvusClient
# from llm_service import DimensionMiningWithQwen

# class Config:
#     DATA_DIR = "./experiment_data"
#     MILVUS_DB = "experiment_data.db"
#     TARGET_COLLECTION = "CmedqaRetrieval_Sampled"
#     PATH_RAW_TAGS = os.path.join(DATA_DIR, "tags_pro_sampled_med.json")
    
#     # 你的维度列表 (保持不变)
#     DIMS_ENUM = ['适宜人群', '适用阶段']
#     DIMS_NUM = ['性能指标', '量化技术指标', '工艺与操作参数']
#     DIMS_DES = [
#         '材料构成', '适配条件', '操作步骤', '病因机制', '治疗方案', 
#         '功效作用', '临床表现', '适用场景', '作用原理', '使用禁忌', 
#         '注意事项', '适用环境', '预防措施', '产品型号', '质量评价属性', 
#         '专业技术术语', '运行设备类型与能力', '运行环境与兼容性'
#     ]
#     ALL_DIMS = DIMS_ENUM + DIMS_NUM + DIMS_DES
    
#     MIN_DOC_COUNT = 5000 

# if not os.path.exists(Config.DATA_DIR):
#     os.makedirs(Config.DATA_DIR)

# class LabelGenerator:
#     def __init__(self):
#         self.miner = DimensionMiningWithQwen()
#         self.client = MilvusClient(uri=Config.MILVUS_DB, keep_alive_permit_without_calls=False)

#     def extract_all_tags(self):
#         print(f">>> [Step 3] 开始全量标签生成 (带标准化)...")
        
#         processed_ids = set()
#         results = {}
#         if os.path.exists(Config.PATH_RAW_TAGS):
#             with open(Config.PATH_RAW_TAGS, 'r', encoding='utf-8') as f:
#                 results = json.load(f)
#             processed_ids = set(results.keys())
#             print(f"检测到断点，已处理 {len(processed_ids)} 条。")

#         try:
#             iterator = self.client.query_iterator(
#                 collection_name=Config.TARGET_COLLECTION,
#                 output_fields=["id", "text"],
#                 batch_size=50 
#             )
#         except Exception as e:
#             print(f"初始化迭代器失败: {e}")
#             return
        
#         save_interval = 50
#         count = 0
        
#         try:
#             while True:
#                 batch = iterator.next()
#                 if not batch: break
                
#                 # 使用 tqdm 显示进度
#                 for doc in tqdm(batch, desc="标准化抽取中"):
#                     doc_id = doc['id']
#                     text = doc['text']
                    
#                     if doc_id in processed_ids: continue
                    
#                     doc_tags = {}
                    
#                     if len(text) > 10:
#                         # 1. 调用新的标准化 Prompt
#                         extracted_dict = self.miner.extract_batch_dimensions(text, Config.ALL_DIMS)
                        
#                         has_valid_content = False
#                         for dim, val_list in extracted_dict.items():
#                             if val_list is not None and len(val_list) > 0:
#                                 doc_tags[dim] = val_list
#                                 has_valid_content = True
                        
#                         # 2. 兜底策略 (Fallback)
#                         if not has_valid_content:
#                             keywords = self.miner.extract_keywords_fallback(text)
#                             if keywords:
#                                 # 这里的 keywords 是一个 list
#                                 doc_tags["补充关键词"] = keywords 
                    
#                     results[doc_id] = doc_tags
#                     processed_ids.add(doc_id)
#                     count += 1
                    
#                     if count % save_interval == 0:
#                         self._save_json(results)
                        
#         except Exception as e:
#             print(f"抽取过程中断: {e}")
#         finally:
#             self._save_json(results)
#             print("抽取结束。")

#     def _save_json(self, data):
#         with open(Config.PATH_RAW_TAGS, 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":    
#     generator = LabelGenerator()
#     generator.extract_all_tags()

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
try:
    from pymilvus import MilvusClient
except ImportError:
    MilvusClient = None
try:
    from .llm_service import DimensionMiningWithQwen
    from .schema_v2 import (
        SCHEMA_VERSION,
        canonicalize_tag_details,
        dimension_path,
        load_schema,
        schema_maps,
    )
except ImportError:
    from llm_service import DimensionMiningWithQwen
    from schema_v2 import (
        SCHEMA_VERSION,
        canonicalize_tag_details,
        dimension_path,
        load_schema,
        schema_maps,
    )
try:
    from .sqlite_text_store import TextStore
except ImportError:
    from sqlite_text_store import TextStore
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class Config:
    DATA_DIR = "./experiment_data"
    MILVUS_DB = "experiment_data.db"
    TARGET_COLLECTION = "CmedqaRetrieval_Sampled"
    SQLITE_PATH = "texts.db"
    QUERY_TABLE_NAME = "CmedqaQueries" 
    DOC_TABLE_NAME = "CmedqaRetrieval"
    
    # 关联关系文件 (用于获取所有需要抽取的 Doc IDs)
    PATH_QRELS = os.path.join(DATA_DIR, "sampled_qrels_med.json")
    
    # 抽取结果保存路径 (如果之前是 tags_debug_top10.json，你可以继续用它，或者改名)
    PATH_TAGS_OUTPUT = os.path.join(DATA_DIR, "tags_debug_top10.json")
    
    # 维度列表
    DIMS_ENUM = ['适宜人群', '适用阶段']
    DIMS_NUM = ['性能指标', '量化技术指标', '工艺与操作参数']
    DIMS_DES =[
        '材料构成', '适配条件', '操作步骤', '疾病类别', '病因机制', '治疗方案', 
        '功效作用', '临床表现', '适用场景', '作用原理', '注意事项', 
        '适用环境', '预防措施', '产品型号', '质量评价属性', 
        '专业技术术语', '运行设备类型与能力', '运行环境与兼容性'
    ]
    ALL_DIMS = DIMS_ENUM + DIMS_NUM + DIMS_DES


def build_tag_document_v2(doc_id, extracted, schema):
    """Normalize one document to array-valued v2 tags plus evidence."""

    schema = load_schema(schema, allow_legacy=False, strict=False)
    tags, details = canonicalize_tag_details(extracted or {}, schema)
    maps = schema_maps(schema)
    paths = sorted({
        " / ".join(maps["nodes"][part]["name"] for part in maps["paths"][dimension_id])
        for dimension_id in tags
    })
    return {
        "tags": tags,
        "tag_details": details,
        "dimension_paths": paths,
        "schema_version": SCHEMA_VERSION,
    }


class TagGeneratorV2:
    """Schema-v2 tag generator; the legacy generator below is unchanged."""

    def __init__(self, schema, miner=None, max_dimensions_per_request=15):
        self.schema = load_schema(schema, allow_legacy=False, strict=False)
        self.miner = miner or DimensionMiningWithQwen()
        self.max_dimensions_per_request = max(1, int(max_dimensions_per_request))

    def run(self, records, output_path=None):
        records = list(records or [])
        leaves = [
            node for node in self.schema["dimensions"]
            if node.get("level") == 2 and node.get("indexable") is True
        ]
        documents = {}
        for record in records:
            doc_id = str(record.get("doc_id", record.get("id", "")))
            text = str(record.get("doc_text", record.get("text", "")) or "")
            extracted = self.miner.extract_batch_dimensions_v2(
                text,
                leaves,
                max_dimensions_per_request=self.max_dimensions_per_request,
            ) if text else {}
            documents[doc_id] = build_tag_document_v2(doc_id, extracted, self.schema)

        result = {"schema_version": SCHEMA_VERSION, "documents": documents}
        if output_path:
            output_path = str(output_path)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        return result

class LabelGeneratorFull:
    def __init__(self):
        print("初始化标签生成器...")
        self.miner = DimensionMiningWithQwen()
        self.doc_store = TextStore(db_path=Config.SQLITE_PATH, table_name=Config.DOC_TABLE_NAME)
        self.results = {}
        self.processed_ids = set()
        
        # 1. 加载历史进度 (断点续传)
        if os.path.exists(Config.PATH_TAGS_OUTPUT):
            print(f"发现已有结果文件 {Config.PATH_TAGS_OUTPUT}，正在加载历史进度...")
            with open(Config.PATH_TAGS_OUTPUT, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            self.processed_ids = set(self.results.keys())
            print(f"-> 已加载 {len(self.processed_ids)} 篇文档的历史标签。")

    def run_extraction(self):
        # 2. 读取 Qrels，获取【所有】目标 Doc IDs
        if not os.path.exists(Config.PATH_QRELS):
            print(f"错误: 找不到 Qrels 文件 {Config.PATH_QRELS}")
            return

        with open(Config.PATH_QRELS, 'r', encoding='utf-8') as f:
            qrels = json.load(f)
        
        target_doc_ids = set()
        for qid, pids in qrels.items():
            if isinstance(pids, list):
                for pid in pids: target_doc_ids.add(str(pid))
            elif isinstance(pids, dict):
                for pid in pids.keys(): target_doc_ids.add(str(pid))
                
        target_doc_ids = list(target_doc_ids)
        print(f"Qrels 中共涉及 {len(target_doc_ids)} 篇正样本文档。")

        # 3. 过滤掉已经处理过的 ID
        remaining_ids =[doc_id for doc_id in target_doc_ids if doc_id not in self.processed_ids]
        print(f"过滤已处理文档后，剩余需要抽取的文档数: {len(remaining_ids)}")
        
        if not remaining_ids:
            print("所有文档均已抽取完毕！")
            return

        # 4. 逐个提取并保存
        save_interval = 50 # 每处理 50 个保存一次
        count = 0
        
        # 使用 tqdm 监控剩余任务的进度
        for doc_id in tqdm(remaining_ids, desc="标签抽取中"):
            text = self.doc_store.get_text(doc_id)
            
            if not text:
                print(f"[Warning] 数据库中未找到文档 ID: {doc_id}，已跳过。")
                # 记录为空，防止下次再查
                self.results[doc_id] = {}
                self.processed_ids.add(doc_id)
                continue

            doc_tags = {}
            if len(text) > 10:
                try:
                    # A. 标准提取
                    extracted_dict = self.miner.extract_batch_dimensions(text, Config.ALL_DIMS)
                    
                    # 过滤空值
                    has_valid_content = False
                    if extracted_dict: # 防止 API 返回 None
                        for dim, val_list in extracted_dict.items():
                            if val_list is not None and len(val_list) > 0:
                                doc_tags[dim] = val_list
                                has_valid_content = True
                    
                    # B. 兜底提取
                    if not has_valid_content:
                        # print(f"\n[ID: {doc_id}] 标准维度未命中，尝试兜底...")
                        keywords = self.miner.extract_keywords_fallback(text)
                        if keywords:
                            doc_tags["其他"] = keywords
                            
                except Exception as e:
                    print(f"\n[Error] 提取文档 {doc_id} 时发生错误: {e}")
                    # 发生错误时不记录到 results 中，这样下次运行还能重试
                    continue
            
            # 将成功抽取(或确定为空)的结果存入内存
            self.results[doc_id] = doc_tags
            self.processed_ids.add(doc_id)
            count += 1
            
            # 5. 定期保存
            if count % save_interval == 0:
                self._save_json()

        # 6. 循环结束，做最后一次保存
        if count % save_interval != 0:
            self._save_json()
            
        print(f"\n✅ 本次运行成功提取了 {count} 篇文档。总计已完成 {len(self.processed_ids)} 篇。")

    def _save_json(self):
        # 静默保存，不打印 log 以免打断 tqdm 进度条
        with open(Config.PATH_TAGS_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generator = LabelGeneratorFull()
    generator.run_extraction()
