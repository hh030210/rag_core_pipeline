import os
import json
try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from scipy.stats import entropy
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # v2 schema utilities remain importable without ML extras
    np = None
    pd = None
    KMeans = None
    pairwise_distances_argmin_min = None
    entropy = None
    cosine_similarity = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from pymilvus import MilvusClient
except ImportError:
    MilvusClient = None
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None

try:
    from .llm_service import DimensionMiningWithQwen
    from .schema_v2 import (
        SCHEMA_VERSION,
        SchemaValidationError,
        as_label_list,
        dimension_path,
        load_schema,
        make_dimension_node,
        normalize_label,
        normalize_text,
        schema_maps,
        stable_dimension_id,
        validate_schema,
    )
except ImportError:
    from llm_service import DimensionMiningWithQwen
    from schema_v2 import (
        SCHEMA_VERSION,
        SchemaValidationError,
        as_label_list,
        dimension_path,
        load_schema,
        make_dimension_node,
        normalize_label,
        normalize_text,
        schema_maps,
        stable_dimension_id,
        validate_schema,
    )

class Config:
    # 路径配置
    DATA_DIR = "./experiment_data"
    MILVUS_DB = "experiment_data.db"
    COLLECTION_NAME = "EcomRetrieval"  # 数据集名称
    
    # 中间结果保存路径
    PATH_V_CAND = os.path.join(DATA_DIR, "V_cand_initial.json")     # 初始候选维度
    PATH_V_CORE = os.path.join(DATA_DIR, "V_core_optimized.json")   # 优化后核心维度
    SCHEMA_VERSION = SCHEMA_VERSION
    PATH_V_CAND_V2 = os.path.join(DATA_DIR, "V_cand_v2.json")
    PATH_V_CORE_V2 = os.path.join(DATA_DIR, "V_core_v2.json")
    PATH_DIM_META_V2 = os.path.join(DATA_DIR, "dimension_metadata_v2.json")
    
    # 算法超参 (参考论文 2.6.3)
    K_CLUSTERS = 50          # 聚类簇数
    N_CORE_SAMPLES = 5       # 质心采样数
    N_BOUND_SAMPLES = 5      # 边界采样数
    
    # 优化阈值 (参考论文 2.6.3)
    TH_COV = 0.50   # 覆盖率阈值
    TH_DIS = 1.0    # 辨识度(熵)阈值 (论文中是15，视标签数量级而定)
    TH_DIFF = 0.3   # 差异性阈值 (R_diff)。如果 R_diff < 0.3，说明相似度 > 0.7，判定为冗余
    MAX_ITER = 3    # 最大迭代轮数·


V2_THRESHOLDS = {
    "leaf_coverage": 0.20,
    "parent_coverage": 0.20,
    "min_support": 2,
    "sibling_label_overlap": 0.70,
    "max_labels_default": 5,
}


def _iter_v2_documents(document_tags):
    if not isinstance(document_tags, dict):
        return []
    if isinstance(document_tags.get("documents"), dict):
        return list(document_tags["documents"].items())
    return list(document_tags.items())


def _document_tag_map(value):
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("tags"), dict):
        return value["tags"]
    return value


def diagnose_leaf_dimensions(document_tags, schema, thresholds=None):
    """Compute leaf coverage/support/overlap diagnostics for v2."""

    schema = load_schema(schema, allow_legacy=False, strict=False)
    maps = schema_maps(schema)
    thresholds = {**V2_THRESHOLDS, **(thresholds or {})}
    docs = _iter_v2_documents(document_tags)
    total = len(docs)
    stats = {}
    for leaf_id in maps["leaves"]:
        labels = []
        covered = 0
        for _doc_id, record in docs:
            values = _document_tag_map(record).get(leaf_id, [])
            values = values if isinstance(values, list) else [values]
            normalized = []
            for value in values:
                label = value.get("label", "") if isinstance(value, dict) else value
                key = normalize_label(label)
                if key and key not in normalized:
                    normalized.append(key)
            if normalized:
                covered += 1
                labels.extend(normalized)
        stats[leaf_id] = {
            "dimension_id": leaf_id,
            "path": dimension_path(schema, leaf_id),
            "covered_documents": covered,
            "total_documents": total,
            "coverage": round(covered / total, 6) if total else 0.0,
            "min_support": len(set(labels)),
            "min_support_threshold": int(thresholds["min_support"]),
            "min_support_pass": len(set(labels)) >= int(thresholds["min_support"]),
            "unique_labels": len(set(labels)),
            "average_labels_per_document": round(len(labels) / covered, 6) if covered else 0.0,
            "coverage_threshold": float(thresholds["leaf_coverage"]),
            "coverage_pass": (covered / total >= float(thresholds["leaf_coverage"])) if total else False,
            "indexable": True,
        }

    for parent_id, child_ids in maps["children"].items():
        child_sets = {child: set() for child in child_ids}
        for _doc_id, record in docs:
            tag_map = _document_tag_map(record)
            for child in child_ids:
                for value in tag_map.get(child, []) if isinstance(tag_map.get(child, []), list) else [tag_map.get(child, [])]:
                    label = value.get("label", "") if isinstance(value, dict) else value
                    key = normalize_label(label)
                    if key:
                        child_sets[child].add(key)
        overlaps = []
        for index, left in enumerate(child_ids):
            for right in child_ids[index + 1:]:
                union = child_sets[left] | child_sets[right]
                overlaps.append(len(child_sets[left] & child_sets[right]) / len(union) if union else 0.0)
        for child in child_ids:
            stats[child]["sibling_label_overlap"] = round(sum(overlaps) / len(overlaps), 6) if overlaps else 0.0
            stats[child]["sibling_label_overlap_threshold"] = float(thresholds["sibling_label_overlap"])
            stats[child]["sibling_label_overlap_high"] = stats[child]["sibling_label_overlap"] >= float(thresholds["sibling_label_overlap"])
        stats[parent_id] = {
            "dimension_id": parent_id,
            "path": dimension_path(schema, parent_id),
            "covered_documents": sum(
                1 for _doc_id, record in docs
                if any(_document_tag_map(record).get(child) for child in child_ids)
            ),
            "total_documents": total,
            "coverage": round(
                sum(1 for _doc_id, record in docs if any(_document_tag_map(record).get(child) for child in child_ids)) / total,
                6,
            ) if total else 0.0,
            "coverage_threshold": float(thresholds["parent_coverage"]),
            "coverage_pass": (
                sum(1 for _doc_id, record in docs if any(_document_tag_map(record).get(child) for child in child_ids)) / total
                >= float(thresholds["parent_coverage"])
            ) if total else False,
            "indexable": False,
            "child_count": len(child_ids),
            "child_ids": child_ids,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": thresholds,
        "total_documents": total,
        "dimensions": stats,
    }


def apply_structured_optimization(schema, source_id, decision):
    """Apply KEEP/DELETE/RENAME/SPLIT/MERGE using structured nodes only."""

    schema = load_schema(schema, allow_legacy=False, strict=False)
    nodes = [dict(node) for node in schema["dimensions"]]
    by_id = {node["id"]: node for node in nodes}
    source = by_id.get(source_id)
    if not source:
        raise KeyError(f"未知维度: {source_id}")
    action = str((decision or {}).get("action", "KEEP")).upper()
    new_nodes = (decision or {}).get("new_nodes", [])
    if not isinstance(new_nodes, list):
        raise SchemaValidationError(["优化决策必须使用 new_nodes 数组"])

    def normalized_new_node(raw, default_parent):
        if not isinstance(raw, dict) or not normalize_text(raw.get("name", "")):
            raise SchemaValidationError(["new_nodes 中每个节点必须包含 name"])
        parent_id = normalize_text(raw.get("parent_id")) or default_parent
        return make_dimension_node(
            name=raw["name"],
            parent_id=parent_id,
            level=2,
            description=raw.get("description", ""),
            indexable=True,
            aliases=raw.get("aliases", []),
            max_labels=raw.get("max_labels", 5),
            dimension_id=normalize_text(raw.get("id")) or stable_dimension_id(raw["name"], parent_id),
        )

    if action == "KEEP":
        return schema
    if action == "DELETE":
        nodes = [node for node in nodes if node["id"] != source_id]
    elif action in {"RENAME", "SPLIT", "MERGE"}:
        parent_id = source.get("parent_id")
        if source.get("level") == 1 and action in {"RENAME", "MERGE"}:
            raise SchemaValidationError(["一级分组只能通过 SPLIT 保留父节点并生成叶子，不能按叶子方式重命名或合并"])
        if action == "RENAME" and len(new_nodes) != 1:
            raise SchemaValidationError(["RENAME 必须返回一个 new_nodes 节点"])
        if action == "SPLIT" and len(new_nodes) < 2:
            raise SchemaValidationError(["SPLIT 必须返回至少两个 new_nodes 节点"])
        if action == "MERGE" and len(new_nodes) != 1:
            raise SchemaValidationError(["MERGE 必须返回一个合并后的 new_nodes 节点"])
        if source.get("level") == 1 and action == "SPLIT":
            # Keep the parent and attach all generated leaves to it.
            source["indexable"] = False
            parent_id = source_id
        else:
            nodes = [node for node in nodes if node["id"] != source_id]
        generated = [normalized_new_node(raw, parent_id) for raw in new_nodes]
        nodes.extend(generated)
    else:
        raise SchemaValidationError([f"不支持的优化动作: {action}"])

    result = {"schema_version": SCHEMA_VERSION, "dimensions": nodes}
    errors = validate_schema(result, strict=True)
    if errors:
        raise SchemaValidationError(errors)
    return result

if not os.path.exists(Config.DATA_DIR):
    os.makedirs(Config.DATA_DIR)

# ================= 核心模块 =================

class DimensionMiningSystem:
    def __init__(self):
        self.client = MilvusClient(
            uri=Config.MILVUS_DB,
            token="", 
            timeout=300, # 全局超时也设大一点
            keep_alive_time=30.0,
            keep_alive_timeout=10.0,
            keep_alive_permit_without_calls=False
        )
        self.miner = DimensionMiningWithQwen()
        # 获取当前文件的目录
        current_dir = os.path.dirname(__file__)
        # 构建模型的相对路径
        embedding_model_path = os.path.join(current_dir, 'bge-m3')
        self.encoder = BGEM3FlagModel(embedding_model_path, use_fp16=True, device='cuda') 
    
    def _load_vectors_and_texts(self, limit=10000):
        """
        使用迭代器 (Iterator) 加载 ID, Text 和 Vector
        优势：解决深度分页(Deep Pagination)导致的越查越慢和超时问题，稳定加载全量数据。
        """
        print(f"正在加载数据 ({Config.COLLECTION_NAME}) - 迭代器模式...")
        
        all_ids = []
        all_texts = []
        all_vectors = []
        
        try:
            # 1. 初始化迭代器
            # batch_size 控制每次从服务端拉多少条。
            # 因为包含 text (长文本) 和 vector (1024维)，单次包较大，建议设为 100-200。
            iterator = self.client.query_iterator(
                collection_name=Config.COLLECTION_NAME,
                filter="",  # 匹配所有数据
                output_fields=["id", "text", "vector"],
                batch_size=100  # 稳健设置
            )
            
            # 2. 循环拉取数据
            # tqdm 的 total 设为 limit，用于显示进度
            with tqdm(total=limit, desc="迭代加载数据") as pbar:
                while True:
                    # 获取下一批数据
                    res = iterator.next()
                    
                    if not res:
                        break # 数据取完了
                    
                    # 解析数据
                    for item in res:
                        all_ids.append(item["id"])
                        all_texts.append(item["text"])
                        all_vectors.append(item["vector"])
                    
                    # 更新进度条
                    pbar.update(len(res))
                    
                    # 达到限制数量则停止
                    if len(all_ids) >= limit:
                        print(f"已达到限制数量 {limit}，停止加载。")
                        break
                        
        except Exception as e:
            print(f"迭代器加载中断: {e}")
            # 如果中间出错，尽量返回已获取的数据
            if not all_ids:
                return [], [], []

        # 截断多余的数据 (如果 iterator 返回了比 limit 多一点的数据)
        all_ids = all_ids[:limit]
        all_texts = all_texts[:limit]
        all_vectors = all_vectors[:limit]

        print(f"加载完成，共获取 {len(all_ids)} 条数据。")
        
        # 转换为 numpy 数组
        vectors_np = np.array(all_vectors, dtype=np.float32)
        
        return all_ids, all_texts, vectors_np

    def _load_vectors_only(self, limit=10000):
        """
        使用迭代器 (Iterator) 加载向量
        优势：解决深度分页(Deep Pagination)导致的越查越慢和超时问题
        """
        print(f"正在加载向量数据 (迭代器模式)...")
        
        all_ids = []
        all_vectors = []
        
        # 1. 初始化迭代器
        # batch_size 控制每次从服务端拉多少条，设置为 1000 比较合适
        try:
            iterator = self.client.query_iterator(
                collection_name=Config.COLLECTION_NAME,
                filter="",  # 匹配所有
                output_fields=["id", "vector"],
                batch_size=500  # 每次拉取 500 条
            )
            
            # 2. 循环拉取
            # tqdm 的 total 设为 limit，只为了显示进度条，不影响逻辑
            with tqdm(total=limit, desc="迭代加载向量") as pbar:
                while True:
                    # 获取下一批数据
                    res = iterator.next()
                    
                    if not res:
                        print("数据已全部取完。")
                        break
                    
                    for item in res:
                        all_ids.append(item["id"])
                        all_vectors.append(item["vector"])
                    
                    pbar.update(len(res))
                    
                    # 达到限制数量则停止
                    if len(all_ids) >= limit:
                        print(f"已达到限制数量 {limit}，停止加载。")
                        break
                        
        except Exception as e:
            print(f"迭代器加载中断: {e}")
        
        # 截断多余的数据
        all_ids = all_ids[:limit]
        all_vectors = all_vectors[:limit]

        print(f"向量加载完成，共 {len(all_ids)} 条。")
        vectors_np = np.array(all_vectors, dtype=np.float32)
        return all_ids, vectors_np

    def _fetch_texts_by_ids(self, target_ids):
        """
        阶段二：根据 ID 列表回查 Text
        """
        if not target_ids: return []
        
        print(f"正在回查 {len(target_ids)} 篇核心文档的文本内容...")
        texts = []
        
        # Milvus 的 filter 表达式长度有限制，建议分批查询
        # 比如每次查 100 个 ID
        batch_size = 100
        
        for i in range(0, len(target_ids), batch_size):
            batch_ids = target_ids[i : i + batch_size]
            
            # 构造 filter 表达式: id in ["1", "2", ...]
            # 注意：如果 ID 是字符串，需要加引号；如果是整数则不需要
            # 这里假设 ID 是字符串 (因为之前 Milvus 建表时设为了 string)
            ids_str = str(batch_ids).replace("'", '"') # 确保使用双引号兼容性更好
            expr = f'id in {ids_str}'
            
            try:
                res = self.client.query(
                    collection_name=Config.COLLECTION_NAME,
                    filter=expr,
                    output_fields=["text"], # 【核心修改】只查 text
                    timeout=30
                )
                for item in res:
                    texts.append(item["text"])
            except Exception as e:
                print(f"回查文本失败: {e}")
                
        return texts

    # --- 2.4.2 基于语义向量的聚类采样 (逻辑更新) ---
    def step1_clustering_sampling(self):
        print("\n>>> [Phase 1] 语义聚类采样 (优化版)...")
        
        # 1. 先只拉取向量 (轻量级)
        all_ids, vectors = self._load_vectors_only(limit=10000)
        
        if len(vectors) < Config.K_CLUSTERS:
            print("数据量过少，直接全量回查。")
            return self._fetch_texts_by_ids(all_ids)

        # 2. KMeans 聚类 (纯内存计算)
        print(f"正在聚类 {len(vectors)} 条向量...")
        kmeans = KMeans(n_clusters=Config.K_CLUSTERS, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        selected_indices = set()
        
        # 3. 质心+边界采样 (计算索引)
        for i in range(Config.K_CLUSTERS):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0: continue
            
            cluster_vectors = vectors[cluster_indices]
            center = kmeans.cluster_centers_[i].reshape(1, -1)
            
            dists = pairwise_distances_argmin_min(cluster_vectors, center, metric='euclidean')[1]
            
            # 找索引
            core_local_idx = np.argsort(dists)[:Config.N_CORE_SAMPLES]
            bound_local_idx = np.argsort(dists)[-Config.N_BOUND_SAMPLES:]
            
            # 将局部索引转换为全局索引
            for idx in np.concatenate([core_local_idx, bound_local_idx]):
                global_idx = cluster_indices[idx]
                selected_indices.add(global_idx)
        
        # 4. 映射回 ID
        target_ids = [all_ids[idx] for idx in selected_indices]
        print(f"采样完成，锁定 {len(target_ids)} 个核心 ID。")
        
        # 5. 最后再去 Milvus 查文本 (按需查询)
        sampled_docs = self._fetch_texts_by_ids(target_ids)
        
        print(f"文本回查完成，获取 {len(sampled_docs)} 篇文档。")
        return sampled_docs

    # --- v2 层次化 Schema API ---
    def step2_generate_candidates_v2(self, sampled_docs, output_path=None):
        """Generate ``V_cand_v2.json`` without touching legacy artifacts."""

        output_path = output_path or Config.PATH_V_CAND_V2
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as handle:
                return load_schema(json.load(handle), allow_legacy=False)
        schema = self.miner.generate_candidate_schema(sampled_docs)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(schema, handle, ensure_ascii=False, indent=2)
        return schema

    @staticmethod
    def diagnose_v2(document_tags, schema, thresholds=None):
        return diagnose_leaf_dimensions(document_tags, schema, thresholds=thresholds)

    @staticmethod
    def apply_v2_decision(schema, source_id, decision):
        return apply_structured_optimization(schema, source_id, decision)

    # --- 2.4.3 初始维度生成 ---
    def step2_generate_candidates(self, sampled_docs):
        if os.path.exists(Config.PATH_V_CAND):
            print("检测到已有候选维度，跳过生成。")
            with open(Config.PATH_V_CAND, 'r') as f:
                return json.load(f)

        print(">>> [Phase 2] LLM 归纳候选维度...")
        
        V_cand = self.miner.generate_candidate_dimensions(sampled_docs)
        
        # 保存中间结果
        with open(Config.PATH_V_CAND, 'w', encoding='utf-8') as f:
            json.dump(V_cand, f, ensure_ascii=False, indent=2)
        print(f"生成初始维度集合 V_cand (Size: {len(V_cand)})")
        return V_cand

    # 辅助函数：计算 Jaccard 系数 (公式 2-6)
    def _calc_jaccard(self, list_a, list_b):
        set_a = {x for x in list_a if x and x != "NULL"}
        set_b = {x for x in list_b if x and x != "NULL"}
        if not set_a and not set_b: return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    # --- 2.4.4 & 2.4.5 迭代优化 ---
    def step3_iterative_optimization(self, initial_dims):
        if os.path.exists(Config.PATH_V_CORE):
            print("检测到已有核心维度，跳过优化。")
            with open(Config.PATH_V_CORE, 'r') as f:
                return json.load(f)

        print(">>> [Phase 3] 维度迭代优化...")
        current_dims = initial_dims
        
        # 加载验证集（扩大样本量以更准确评估覆盖率）
        ids, texts, _ = self._load_vectors_and_texts(limit=500)
        # validation_texts = texts[:20] 

        # 白名单，记录已经被判定为 KEEP 的维度，不再重复计算
        verified_dims = set()

        for iteration in range(Config.MAX_ITER):
            print(f"\n--- Iteration {iteration + 1} / {Config.MAX_ITER} ---")
            
            # 1. 抽取逻辑 (保持不变)
            active_dims = [d for d in current_dims if d not in verified_dims]
            if not active_dims: break
            
            extraction_results = {dim: [] for dim in active_dims}
            
            # 使用 tqdm 显示进度
            for text in tqdm(texts, desc="抽取进度"):
                for dim in active_dims:
                    res = self.miner.extract_dimension_value(text, dim)
                    if res is not None:
                        extraction_results[dim].append(res)
            
            # 用于标记本轮要删除的维度
            dims_to_remove = set()
            new_dims_added = []
            
            # =================================================
            # Part A: 单维度诊断 (覆盖率 & 辨识度)
            # =================================================
            # 1) 先扫一遍, 划分"合格维度"与"不合格维度"
            candidates_for_diff_check = []
            failing_dims = []   # (dim, cov, dis, issue_type, issue_msg, sample_values)

            for dim in active_dims:
                values = extraction_results[dim]

                cov = len(values) / len(texts)
                if len(values) > 0:
                    value_counts = Counter(values)
                    probs = [c / len(values) for c in value_counts.values()]
                    dis = entropy(probs)
                else:
                    dis = 0.0

                if cov < Config.TH_COV:
                    issue_type = "低覆盖率"
                    issue_msg = f"覆盖率仅 {cov:.2%} (阈值 {Config.TH_COV:.2%})"
                    failing_dims.append((dim, cov, dis, issue_type, issue_msg, values))
                elif dis < Config.TH_DIS:
                    issue_type = "低辨识度"
                    issue_msg = f"信息熵 {dis:.2f} (阈值 {Config.TH_DIS:.2f}, 辨识度低)"
                    failing_dims.append((dim, cov, dis, issue_type, issue_msg, values))
                else:
                    candidates_for_diff_check.append(dim)

            # =================================================
            # Part A1: 不合格维度的"分阶段融合"流程
            # 规则:
            #   Stage 1 -> 尝试与"合格维度"融合
            #   Stage 2 -> 失败后再尝试与"其他不合格维度"融合
            #   兜底   -> 两阶段都失败 -> 直接 DELETE (不再强行融合, 也不再交 LLM 自选)
            # =================================================
            for dim, cov, dis, issue_type, issue_msg, values in failing_dims:
                merged = False

                # ---- Stage 1: 与"合格维度"融合 ----
                if candidates_for_diff_check:
                    decision_s1 = self.miner.merge_with_targets(
                        dim_name=dim,
                        issue_type=issue_type,
                        metric_data=issue_msg,
                        sample_values=values[:5],
                        candidate_targets=candidates_for_diff_check,
                    )
                    if (decision_s1.get("action") or "").upper() == "MERGE":
                        print(f"    [Stage1 融合] {dim} -> {decision_s1.get('merge_target')}")
                        # 复用现有 MERGE 处理: 删除当前 dim, 加入 new_dimensions
                        self._handle_llm_decision(
                            dim,
                            {
                                "action": "MERGE",
                                "reasoning": decision_s1.get("reasoning", ""),
                                "new_dimensions": decision_s1.get("new_dimensions", []),
                            },
                            dims_to_remove,
                            new_dims_added,
                            verified_dims,
                        )
                        merged = True

                # ---- Stage 2: 与"其他不合格维度"融合 ----
                if not merged:
                    other_failing = [d for d, _, _, _, _, _ in failing_dims if d != dim]
                    # 去掉本轮已经被移除的 (避免和已死的维度合)
                    other_failing = [d for d in other_failing if d not in dims_to_remove]
                    if other_failing:
                        decision_s2 = self.miner.merge_with_targets(
                            dim_name=dim,
                            issue_type=issue_type,
                            metric_data=issue_msg,
                            sample_values=values[:5],
                            candidate_targets=other_failing,
                        )
                        if (decision_s2.get("action") or "").upper() == "MERGE":
                            target = decision_s2.get("merge_target", "")
                            print(f"    [Stage2 融合] {dim} -> {target}")
                            self._handle_llm_decision(
                                dim,
                                {
                                    "action": "MERGE",
                                    "reasoning": decision_s2.get("reasoning", ""),
                                    "new_dimensions": decision_s2.get("new_dimensions", []),
                                },
                                dims_to_remove,
                                new_dims_added,
                                verified_dims,
                            )
                            # 同步移除被合并的另一方, 避免出现"新增了合并后的维度但旧方还在"
                            if target and target in other_failing:
                                dims_to_remove.add(target)
                            merged = True

                # ---- 兜底: 两阶段都没融进去 -> 直接 DELETE ----
                if not merged:
                    print(f"    [兜底删除] {dim} (融合失败, 直接删除)")
                    self._handle_llm_decision(
                        dim,
                        {"action": "DELETE", "reasoning": "两阶段融合均失败, 按规则兜底删除", "new_dimensions": []},
                        dims_to_remove,
                        new_dims_added,
                        verified_dims,
                    )

            # =================================================
            # Part B: 双维度差异性评估 (冗余检测)
            # =================================================
            # 仅在合格者之间进行两两比较
            check_list = [d for d in candidates_for_diff_check if d not in dims_to_remove]
            
            if len(check_list) > 1:
                print(f"正在进行差异性评估 (Candidates: {len(check_list)})...")
                # 1. 批量计算语义向量 (Sim_def)
                dim_vectors = self.encoder.encode(check_list, return_dense=True)['dense_vecs']
                sim_def_matrix = cosine_similarity(dim_vectors)
                
                skip_indices = set() # 避免 A和B比较后，B又和A比较

                for i in range(len(check_list)):
                    if i in skip_indices: continue
                    
                    for j in range(i + 1, len(check_list)):
                        if j in skip_indices: continue
                        
                        dim_a = check_list[i]
                        dim_b = check_list[j]
                        
                        # (2-5) Static Similarity
                        sim_def = sim_def_matrix[i][j]
                        
                        # (2-6) Dynamic Data Similarity (Jaccard)
                        vals_a = extraction_results.get(dim_a, [])
                        vals_b = extraction_results.get(dim_b, [])
                        sim_data = self._calc_jaccard(vals_a, vals_b)
                        
                        # (2-7) Independence Score R_diff = 1 - min(...)
                        # 冗余度 = min(Sim_def, Sim_data)
                        redundancy_score = min(sim_def, sim_data)
                        r_diff = 1.0 - redundancy_score
                        
                        # (2-8) 判定冗余
                        if r_diff < Config.TH_DIFF: # 例如小于 0.3，即相似度 > 0.7
                            print(f"  [冗余告警] {dim_a} vs {dim_b} (冗余度: {redundancy_score:.2f})")
                            
                            # 构造输入给 LLM，复用 optimize_dimension
                            # 技巧：把 dim_b 的信息塞进 metric_data 和 sample_values
                            issue_type = "语义/数据冗余"
                            metric_data = f"与维度“{dim_b}”高度重叠 (冗余度 {redundancy_score:.2f})。"
                            
                            # 构造对比样本
                            combined_samples = [f"[本维度]: {v}" for v in vals_a[:3]] + \
                                               [f"[冗余对象-{dim_b}]: {v}" for v in vals_b[:3]]
                            
                            decision = self.miner.optimize_dimension(
                                dim_name=dim_a, # 我们主要优化 A
                                issue_type=issue_type,
                                metric_data=metric_data,
                                sample_values=combined_samples
                            )
                            
                            # 处理决策
                            # 特殊处理：如果是 MERGE，需要同时删除 A 和 B
                            if decision:
                                action = decision.get("action", "").upper()
                                self._handle_llm_decision(dim_a, decision, dims_to_remove, new_dims_added, verified_dims)
                                
                                # 如果动作是 MERGE，意味着 A 和 B 合并成新维度，所以 B 也要删掉
                                if action == "MERGE":
                                    dims_to_remove.add(dim_b)
                                    skip_indices.add(j) # B 已经被处理了，跳过
                                # 如果动作是 DELETE (删除A)，则 B 保留
                                # 如果动作是 KEEP (两个都留)，则 B 保留

            # =================================================
            # 更新集合
            # =================================================
            # 1. 将通过 Part A 且未在 Part B 中被删除的维度加入白名单
            for dim in check_list:
                if dim not in dims_to_remove:
                    verified_dims.add(dim)

            current_dims = [d for d in current_dims if d not in dims_to_remove]
            for nd in new_dims_added:
                nd = nd.strip()
                if nd and nd not in current_dims:
                    current_dims.append(nd)
            
            print(f"本轮结束 -> 移除: {len(dims_to_remove)}, 新增: {len(new_dims_added)}。")
            if len(dims_to_remove) == 0 and len(new_dims_added) == 0:
                print("系统收敛。")
                break
        
        # ... 保存结果 ...
        return current_dims

    # 统一处理 LLM 决策的辅助函数
    def _handle_llm_decision(self, dim, decision, dims_to_remove, new_dims_added, verified_dims):
        action = decision.get("action", "").upper()
        reason = decision.get("reasoning", "")
        new_dims = decision.get("new_dimensions", [])
        if not isinstance(new_dims, list): new_dims = [str(new_dims)] if new_dims else []

        print(f"    决策: {action} | 理由: {reason[:50]}...")

        if action == "DELETE":
            dims_to_remove.add(dim)
        elif action == "KEEP":
            verified_dims.add(dim)
        elif action == "RENAME":
            dims_to_remove.add(dim)
            new_dims_added.extend(new_dims)
        elif action == "SPLIT":
            dims_to_remove.add(dim)
            new_dims_added.extend(new_dims)
        elif action == "MERGE":
            # 这里的 MERGE 意味着：删除当前 dim (以及外部逻辑删除 dim_b)，添加新维度
            dims_to_remove.add(dim)
            new_dims_added.extend(new_dims)

# ================= 主程序入口 =================

if __name__ == "__main__":
    # 初始化系统
    system = DimensionMiningSystem()
    
    # Step 1: 聚类采样 (获取代表性文档)
    sampled_docs = system.step1_clustering_sampling()
    
    # Step 2: 初始维度生成 (LLM 归纳)
    # 结果保存至: experiment_data/V_cand_{Collection}.json
    V_cand = system.step2_generate_candidates(sampled_docs)
    
    # Step 3: 维度迭代优化 (LLM 反思 + 数据验证)
    # 结果保存至: experiment_data/V_core_{Collection}.json
    V_core = system.step3_iterative_optimization(V_cand)
    
    print("\n" + "="*50)
    print("维度挖掘流程结束！")
    print(f"初始候选维度数: {len(V_cand)}")
    print(f"最终核心维度数: {len(V_core)}")
    print(f"最终维度列表: {V_core}")
    print("="*50)
