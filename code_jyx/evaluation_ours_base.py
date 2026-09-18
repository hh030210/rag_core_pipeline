import os
import sys  # 新增
import json
import pickle
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import contextmanager 

from index_builder import IndexConfig
from query_parser import QueryParser
from sqlite_text_store import TextStore

# --- 新增：定义一个用于屏蔽 tqdm 内部输出的上下文管理器 ---
@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

class RetrievalConfig:
    DATA_DIR = "./experiment_data"
    MILVUS_DB = "experiment_data.db"
    TARGET_COLLECTION = "CmedqaRetrieval_Sampled"
    
    PATH_QRELS = os.path.join(DATA_DIR, "sampled_qrels_med.json")
    
    # 保存 Query 解析结果的缓存文件路径
    PATH_PARSED_QUERIES = os.path.join(DATA_DIR, "parsed_queries_cache.json")

    # SQLite 配置
    SQLITE_PATH = "texts.db"
    QUERY_TABLE_NAME = "CmedqaQueries" 
    DOC_TABLE_NAME = "CmedqaRetrieval"
    
    # 实验指标设置
    RECALL_KS = [5, 10, 20, 50, 100] 
    VECTOR_TOP_K = 200 

class MetricsCalculator:
    @staticmethod
    def calc_recall_at_k(retrieved_ids, truth_ids, k):
        if not truth_ids: return 0.0
        cut_retrieved = set(retrieved_ids[:k])
        hit_count = len(cut_retrieved & set(truth_ids))
        return hit_count / len(truth_ids)

class CascadeRetrievalExperiment:
    def __init__(self):
        print(">>> 初始化 Recall 导向实验环境...")
        self.client = MilvusClient(uri=RetrievalConfig.MILVUS_DB)
        self.parser = QueryParser()
        self.metrics = MetricsCalculator()
        
        self.query_store = TextStore(db_path=RetrievalConfig.SQLITE_PATH, table_name=RetrievalConfig.QUERY_TABLE_NAME)
        self.doc_store = TextStore(db_path=RetrievalConfig.SQLITE_PATH, table_name=RetrievalConfig.DOC_TABLE_NAME)

        print("加载索引文件...")
        with open(IndexConfig.PATH_INVERTED_INDEX, 'r') as f:
            self.inverted_index = json.load(f)
        
        # -------------------- 构建正向索引 (Doc ID -> Tags) --------------------
        print("正在从倒排索引构建文档标签正向索引...")
        self.doc_tags = {}
        for dim, tag_docs in self.inverted_index.items():
            for tag_val, doc_ids in tag_docs.items():
                for d_id in doc_ids:
                    d_id_str = str(d_id)
                    if d_id_str not in self.doc_tags:
                        self.doc_tags[d_id_str] = {}
                    if dim not in self.doc_tags[d_id_str]:
                        self.doc_tags[d_id_str][dim] = []
                    self.doc_tags[d_id_str][dim].append(tag_val)
        # -------------------------------------------------------------------------
        
        with open(IndexConfig.PATH_DIM_META, 'r') as f:
            self.dim_meta = json.load(f)


        self.tag_vectors = {}
        if os.path.exists(IndexConfig.PATH_TAG_VECTORS):
            with open(IndexConfig.PATH_TAG_VECTORS, 'rb') as f:
                self.tag_vectors = pickle.load(f)
        
        # 加载已缓存的解析结果
        self.parsed_queries_cache = {}
        self.cache_updated = False # 标记本次运行是否产生了新的解析结果
        if os.path.exists(RetrievalConfig.PATH_PARSED_QUERIES):
            print(f"加载 Query 解析结果缓存 ({RetrievalConfig.PATH_PARSED_QUERIES})...")
            with open(RetrievalConfig.PATH_PARSED_QUERIES, 'r', encoding='utf-8') as f:
                self.parsed_queries_cache = json.load(f)

        print("加载 BGE-M3 模型...")
        current_dir = os.path.dirname(__file__)
        embedding_model_path = os.path.join(current_dir, 'bge-m3')
        self.encoder = BGEM3FlagModel(embedding_model_path, use_fp16=True, device='cuda')
        
    def _match_open_tags(self, dim, raw_val):
        if dim not in self.tag_vectors: return None
        target_data = self.tag_vectors[dim]
        # --- 屏蔽编码时的刷屏 ---
        with suppress_stderr():
            query_vec = self.encoder.encode([raw_val], return_dense=True)['dense_vecs']
        scores = cosine_similarity(query_vec, target_data['vectors'])[0]
        best_idx = np.argmax(scores)
        if scores[best_idx] > 0.8: return target_data['values'][best_idx]
        return None

    def _get_soft_matched_tags(self, dim, raw_val, threshold=0.65):
        """
        [新增] 软映射：返回所有相似度高于阈值的标准标签及其相似度得分
        返回格式: {"婴幼儿": 0.85, "儿童": 0.72}
        """
        if dim not in self.tag_vectors: return {}
        target_data = self.tag_vectors[dim]
        
        with suppress_stderr():
            query_vec = self.encoder.encode([raw_val], return_dense=True)['dense_vecs']
            
        scores = cosine_similarity(query_vec, target_data['vectors'])[0]
        
        matched_tags = {}
        # 找出所有大于阈值的标签
        for idx, score in enumerate(scores):
            if score >= threshold:
                matched_tags[target_data['values'][idx]] = float(score)
                
        return matched_tags

    def _get_tag_matching_docs(self, query_constraints, qid):
        """
        [优化版] 向量化软匹配
        返回: {doc_id: float_score} 
        """
        if not query_constraints: return {}, {}
            
        doc_tag_scores = {}
        debug_logs =[]
        doc_tag_evidence = {} # [新增] 记录每个文档的加分依据

        excluded_dims = {"适宜人群", "适用阶段"}
        
        for dim, vals in query_constraints.items():
            if dim in excluded_dims: 
                continue

            if dim not in self.inverted_index: continue
            
            # 记录当前维度下，每个 doc 的最高得分 (防止同一个维度的多个近义词重复加分)
            dim_doc_scores = {} 
            dim_doc_evidence = {}
            
            for v in vals:
                # 1. 枚举型：严格匹配，命中得 1.0 分
                if self.dim_meta[dim]['is_enum']:
                    if v in self.inverted_index[dim]:
                        for doc_id in self.inverted_index[dim][v]:
                            dim_doc_scores[doc_id] = max(dim_doc_scores.get(doc_id, 0.0), 1.0)
                # 2. 开放型：向量软匹配
                else:
                    # 获取一组相似标签及其分数，如 {"腹痛": 0.9, "胃痛": 0.68}
                    matched_tags_with_scores = self._get_soft_matched_tags(dim, v, threshold=0.65)
                    
                    # 兜底：如果正好有字面完全一致的，强制给 1.0
                    if v in self.inverted_index[dim] and v not in matched_tags_with_scores:
                        matched_tags_with_scores[v] = 1.0
                        
                    if matched_tags_with_scores:
                        log_str = ", ".join([f"'{tk}'({sc:.2f})" for tk, sc in matched_tags_with_scores.items()])
                        debug_logs.append(f"  [Soft Match] '{v}' -> {log_str} (维度:{dim})")
                    
                    # 把分数赋给对应的文档
                    for tag_val, sim_score in matched_tags_with_scores.items():
                        for doc_id in self.inverted_index[dim][tag_val]:
                            # 同一个维度，取能匹配到的最大分数
                            dim_doc_scores[doc_id] = max(dim_doc_scores.get(doc_id, 0.0), sim_score)
                            dim_doc_evidence[doc_id] = f"[{dim}] Query中包含'{v}' -> 软命中倒排标签 '{tag_val}' (相似度: {sim_score:.2f})"
            
            # 3. 将当前维度的得分累加到全局总得分中
            for doc_id, score in dim_doc_scores.items():
                doc_tag_scores[doc_id] = doc_tag_scores.get(doc_id, 0.0) + score
                if doc_id not in doc_tag_evidence:
                    doc_tag_evidence[doc_id] =[]
                doc_tag_evidence[doc_id].append(dim_doc_evidence[doc_id])

        # 可选：打印匹配日志
        # if debug_logs:
        #     print(f"\n[Query {qid}] 软匹配日志:")
        #     for log in debug_logs: print(log)
            
        return doc_tag_scores, doc_tag_evidence # 返回 {doc_id: 1.85, doc_id2: 0.72, ...}

    # def run(self):
    #     with open(RetrievalConfig.PATH_QRELS, 'r') as f:
    #         qrels = json.load(f)

    #     results_summary = {
    #         "Base": {k: [] for k in RetrievalConfig.RECALL_KS},
    #         "Ours": {k: [] for k in RetrievalConfig.RECALL_KS}
    #     }
        
    #     test_qids = list(qrels.keys())[:100] 
        
    #     for qid in tqdm(test_qids, desc="Evaluating"):
    #         query_text = self.query_store.get_text(str(qid))
    #         if not query_text: continue
            
    #         # print(f"\n{'='*50}\n正在评估 Query: {query_text[:50]}... (ID: {qid})")
    #         truth_docs = set(qrels[qid])
            
    #         # 1. 向量检索
    #         # --- 屏蔽编码时的刷屏 ---
    #         with suppress_stderr():
    #             raw_vec = self.encoder.encode([query_text], return_dense=True)['dense_vecs'][0]
    #         query_vec = raw_vec.astype(np.float32).tolist()
            
    #         res = self.client.search(
    #             collection_name=RetrievalConfig.TARGET_COLLECTION,
    #             data=[query_vec],
    #             limit=RetrievalConfig.VECTOR_TOP_K, 
    #             output_fields=["id"]
    #         )
            
    #         vec_candidates = [(hit['id'], hit['distance']) for hit in res[0]]
    #         vec_candidate_ids_set = set([x[0] for x in vec_candidates])
            
    #         # Base Metrics
    #         base_ranking = [x[0] for x in vec_candidates]
    #         for k in RetrievalConfig.RECALL_KS:
    #             score = self.metrics.calc_recall_at_k(base_ranking, truth_docs, k)
    #             results_summary["Base"][k].append(score)

    #        # 2. 标签解析与匹配 (获取的是软得分字典)
    #         qid_str = str(qid)
    #         if qid_str in self.parsed_queries_cache:
    #             constraints = self.parsed_queries_cache[qid_str]
    #         else:
    #             # 缓存没有，调用大模型/规则解析，并存入字典
    #             constraints = self.parser.parse(qid, query_text, self.dim_meta)
    #             self.parsed_queries_cache[qid_str] = constraints
    #             self.cache_updated = True # 标记为需要落盘

    #         doc_tag_scores, doc_tag_evidence = self._get_tag_matching_docs(constraints, qid)
            
    #         # 3. 软融合排序 (Soft Fusion)
    #         final_ranking_tuples =[]
            
    #         # 调节因子 (超参数)：控制标签对最终排序的干预力度
    #         # BGE-M3 的余弦分数通常在 0.3 ~ 0.8 之间
    #         # Tag Score 可能是 0.0, 0.8, 1.6 等
    #         # 设置 ALPHA = 0.2，意味着一个完美的标签匹配会给总分增加 0.2，相当于把排名稍微往前推，但不至于彻底颠覆
    #         ALPHA = 0.2 

    #          # 将 vec_candidates 转为 dict 方便后面打印具体向量分数
    #         vec_scores_dict = {doc_id: score for doc_id, score in vec_candidates}
            
    #         for doc_id, vec_score in vec_candidates:
    #             # 获取该文档的标签加分，没有就是 0
    #             t_score = doc_tag_scores.get(doc_id, 0.0) if doc_tag_scores else 0.0
                
    #             # 核心融合公式
    #             final_score = vec_score + ALPHA * t_score
                
    #             final_ranking_tuples.append((doc_id, final_score))
                
    #         # 根据综合分数重新排序
    #         final_ranking_tuples.sort(key=lambda x: x[1], reverse=True)
    #         final_ranking = [x[0] for x in final_ranking_tuples]
            
    #         # Ours Metrics 计算
    #         for k in RetrievalConfig.RECALL_KS:
    #             score = self.metrics.calc_recall_at_k(final_ranking, truth_docs, k)
    #             results_summary["Ours"][k].append(score)

    #         # -------------------- [新增] Case Study 对比打印 --------------------
    #         # 找出 Base Recall@10 没有命中，但 Ours Recall@10 命中的文档
    #         base_hits_10 = set(base_ranking[:10]) & truth_docs
    #         ours_hits_10 = set(final_ranking[:10]) & truth_docs
            
    #         # 计算纯粹因为我们算法而"新召回"的文档
    #         improved_docs = ours_hits_10 - base_hits_10
            
    #         if improved_docs:
    #             print(f"\n\n{'='*20} 🎯 发现成功优化的 Case (Recall@10) {'='*20}")
    #             print(f"Query ID: {qid}")
    #             print(f"Query 文本: {query_text}")
    #             print(f"提取的约束: {constraints}")
                
    #             for doc_id in improved_docs:
    #                 base_rank = base_ranking.index(doc_id) + 1 if doc_id in base_ranking else -1
    #                 ours_rank = final_ranking.index(doc_id) + 1
                    
    #                 v_score = vec_scores_dict.get(doc_id, 0.0)
    #                 t_score = doc_tag_scores.get(doc_id, 0.0)
    #                 f_score = v_score + ALPHA * t_score

    #                 doc_text = self.doc_store.get_text(str(doc_id))
    #                 if not doc_text:
    #                     doc_text = "(未能在 SQLite 中查找到该 doc_id 的原文)"
    #                 else:
    #                     # 如果文本太长可以截断，如 doc_text[:150] + "..."
    #                     doc_text = doc_text.replace('\n', ' ') 

    #                 # 获取文档自身的所有结构化标签
    #                 doc_id_str = str(doc_id)
    #                 doc_own_tags = self.doc_tags.get(doc_id_str, {})
    #                 # 格式化拼接，如：[症状] 发热, 胃痛 | [疾病] 肠胃炎
    #                 doc_tags_str = " | ".join([f"[{d}] {', '.join(ts)}" for d, ts in doc_own_tags.items()])
    #                 if not doc_tags_str:
    #                     doc_tags_str = "(无标签或未被抽取)"
                    
    #                 print(f"\n  ➤ 目标召回 Doc ID: {doc_id}")
    #                 print(f"    📈 排名变化: Base Rank {base_rank} -> Ours Rank {ours_rank} (进入前10!)")
    #                 print(f"    🔢 分数构成: 纯向量分 {v_score:.4f} + {ALPHA} * 标签分 {t_score:.4f} = {f_score:.4f}")
    #                 print(f"    📄 文档原文: {doc_text}") 
    #                 print(f"    🏷️ 文档标签: {doc_tags_str}")  # [新增打印]
    #                 print(f"    💡 加分依据 (软标签的作用):")
                    
    #                 evidences = doc_tag_evidence.get(doc_id,[])
    #                 if evidences:
    #                     for ev in evidences:
    #                         print(f"       ✅ {ev}")
    #                 else:
    #                     print("       ⚠️ 异常：没有匹配到标签分，可能是因为其他文档名次下降导致的被动上升。")
    #             print("="*70 + "\n")
    #         # ------------------------------------------------------------------

    #     # -------------------- 循环结束后保存更新的解析结果 --------------------
    #     if self.cache_updated:
    #         print(f"\n保存更新后的 Query 解析结果到 {RetrievalConfig.PATH_PARSED_QUERIES}...")
    #         # 如果目录不存在，顺便建立一下目录，防止报错
    #         os.makedirs(os.path.dirname(RetrievalConfig.PATH_PARSED_QUERIES), exist_ok=True)
    #         with open(RetrievalConfig.PATH_PARSED_QUERIES, 'w', encoding='utf-8') as f:
    #             json.dump(self.parsed_queries_cache, f, ensure_ascii=False, indent=2)
    #     # -----------------------------------------------------------------------------

    #     # 最终结果
    #     print("\n" + "="*50)
    #     print(f"{'Metric':<15} | {'Base':<10} | {'Ours':<10} | {'Lift':<10}")
    #     print("-" * 50)
        
    #     for k in RetrievalConfig.RECALL_KS:
    #         base_avg = np.mean(results_summary["Base"][k])
    #         ours_avg = np.mean(results_summary["Ours"][k])
    #         lift = (ours_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0.0
    #         print(f"Recall@{k:<4}    | {base_avg:.4f}     | {ours_avg:.4f}     | {lift:+.2f}%")
    #     print("="*50)

    def run(self):
        with open(RetrievalConfig.PATH_QRELS, 'r') as f:
            qrels = json.load(f)

        # 记录三种策略的指标：Base(纯向量), Ours_Score(分数相加), Ours_RRF(倒数排名融合)
        results_summary = {
            "Base": {k: [] for k in RetrievalConfig.RECALL_KS},
            "Ours_Score": {k: [] for k in RetrievalConfig.RECALL_KS},
            "Ours_RRF": {k: [] for k in RetrievalConfig.RECALL_KS}
        }
        
        test_qids = list(qrels.keys())[:100] 
        
        for qid in tqdm(test_qids, desc="Evaluating"):
            query_text = self.query_store.get_text(str(qid))
            if not query_text: continue
            
            truth_docs = set(qrels[qid])
            
            # -----------------------------------------------------------------
            # 1. 向量检索 (Base)
            # -----------------------------------------------------------------
            with suppress_stderr():
                raw_vec = self.encoder.encode([query_text], return_dense=True)['dense_vecs'][0]
            query_vec = raw_vec.astype(np.float32).tolist()
            
            res = self.client.search(
                collection_name=RetrievalConfig.TARGET_COLLECTION,
                data=[query_vec],
                limit=RetrievalConfig.VECTOR_TOP_K, 
                output_fields=["id"]
            )
            
            vec_candidates = [(hit['id'], hit['distance']) for hit in res[0]]
            vec_scores_dict = {doc_id: score for doc_id, score in vec_candidates}
            
            # Base Metrics 统计
            base_ranking = [x[0] for x in vec_candidates]
            for k in RetrievalConfig.RECALL_KS:
                score = self.metrics.calc_recall_at_k(base_ranking, truth_docs, k)
                results_summary["Base"][k].append(score)

            # -----------------------------------------------------------------
            # 2. 标签解析与匹配
            # -----------------------------------------------------------------
            qid_str = str(qid)
            if qid_str in self.parsed_queries_cache:
                constraints = self.parsed_queries_cache[qid_str]
            else:
                constraints = self.parser.parse(qid, query_text, self.dim_meta)
                self.parsed_queries_cache[qid_str] = constraints
                self.cache_updated = True 

            doc_tag_scores, doc_tag_evidence = self._get_tag_matching_docs(constraints, qid)
            
            # 构建排名映射字典 (Rank 从 1 开始)
            # 向量排名
            vec_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(vec_candidates)}
            # 标签排名 (按匹配得分降序排列)
            sorted_tag_docs = sorted(doc_tag_scores.items(), key=lambda x: x[1], reverse=True)
            doc_tag_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_tag_docs)}
            
            # -----------------------------------------------------------------
            # 3. 软融合排序 (对比 Score vs RRF)
            # -----------------------------------------------------------------
            final_ranking_score_tuples = []
            final_ranking_rrf_tuples = []
            
            # [超参数]
            ALPHA_SCORE = 0.2  # Score融合权重
            K = 60             # RRF 平滑常数 (业界标准默认值)
            ALPHA_RRF = 1.0    # RRF 标签权重 (量纲统一后，1.0 代表向量与标签 1:1 平权)
            
            # === 策略 A: Ours_Score (实际分数相加) ===
            # 注意：Score 策略通常只在向量召回的 Top-K 候选池内进行重排
            for doc_id, vec_score in vec_candidates:
                t_score = doc_tag_scores.get(doc_id, 0.0) if doc_tag_scores else 0.0
                final_score_val = vec_score + ALPHA_SCORE * t_score
                final_ranking_score_tuples.append((doc_id, final_score_val))
                
            # === 策略 B: Ours_RRF (倒数排名融合) ===
            # 优势：可以拓宽召回池，将向量和标签两路召回的"所有"候选者合并
            all_candidate_ids = set(vec_ranks.keys()).union(set(doc_tag_ranks.keys()))
            
            for doc_id in all_candidate_ids:
                v_rank = vec_ranks.get(doc_id, float('inf'))
                t_rank = doc_tag_ranks.get(doc_id, float('inf'))
                
                # 计算各自的 RRF 分数
                v_rrf_score = 1.0 / (K + v_rank) if v_rank != float('inf') else 0.0
                t_rrf_score = 1.0 / (K + t_rank) if t_rank != float('inf') else 0.0
                
                # RRF 加权总分
                final_score_rrf = v_rrf_score + ALPHA_RRF * t_rrf_score
                final_ranking_rrf_tuples.append((doc_id, final_score_rrf))
                
            # 分别对两种策略进行排序提取
            final_ranking_score_tuples.sort(key=lambda x: x[1], reverse=True)
            final_ranking_score = [x[0] for x in final_ranking_score_tuples]

            final_ranking_rrf_tuples.sort(key=lambda x: x[1], reverse=True)
            final_ranking_rrf = [x[0] for x in final_ranking_rrf_tuples]
            
            # Metrics 统计
            for k in RetrievalConfig.RECALL_KS:
                results_summary["Ours_Score"][k].append(self.metrics.calc_recall_at_k(final_ranking_score, truth_docs, k))
                results_summary["Ours_RRF"][k].append(self.metrics.calc_recall_at_k(final_ranking_rrf, truth_docs, k))

            # -----------------------------------------------------------------
            # 4. Case Study 对比打印 (基于 RRF 策略)
            # -----------------------------------------------------------------
            # base_hits_10 = set(base_ranking[:10]) & truth_docs
            # ours_rrf_hits_10 = set(final_ranking_rrf[:10]) & truth_docs
            # improved_docs = ours_rrf_hits_10 - base_hits_10
            
            # if improved_docs:
            #     print(f"\n\n{'='*20} 🎯 发现成功优化的 Case (RRF Recall@10) {'='*20}")
            #     print(f"Query ID: {qid}")
            #     print(f"Query 文本: {query_text}")
            #     print(f"提取的约束: {constraints}")
                
            #     for doc_id in improved_docs:
            #         base_rank = base_ranking.index(doc_id) + 1 if doc_id in base_ranking else -1
            #         rrf_rank = final_ranking_rrf.index(doc_id) + 1
                    
            #         v_r = vec_ranks.get(doc_id, '未命中')
            #         t_r = doc_tag_ranks.get(doc_id, '未命中')

            #         doc_text = self.doc_store.get_text(str(doc_id))
            #         if not doc_text:
            #             doc_text = "(未能在 SQLite 中查找到该 doc_id 的原文)"
            #         else:
            #             doc_text = doc_text.replace('\n', ' ')[:150] + "..." 

            #         doc_own_tags = self.doc_tags.get(str(doc_id), {})
            #         doc_tags_str = " | ".join([f"[{d}] {', '.join(ts)}" for d, ts in doc_own_tags.items()])
            #         if not doc_tags_str: doc_tags_str = "(无标签)"
                    
            #         print(f"\n  ➤ 目标召回 Doc ID: {doc_id}")
            #         print(f"    📈 排名变化: Base Rank {base_rank} -> RRF Rank {rrf_rank} (进入前10!)")
            #         print(f"    🔢 排名构成: 向量第 {v_r} 名, 标签第 {t_r} 名")
            #         print(f"    📄 文档原文: {doc_text}") 
            #         print(f"    🏷️ 文档标签: {doc_tags_str}")
            #         print(f"    💡 加分依据:")
            #         evidences = doc_tag_evidence.get(doc_id,[])
            #         if evidences:
            #             for ev in evidences: print(f"       ✅ {ev}")
            #         else:
            #             print("       ⚠️ 异常：没有匹配到标签分，可能是因为其他文档名次下降导致的被动上升。")
            #     print("="*75 + "\n")

        # 保存 Cache
        if self.cache_updated:
            os.makedirs(os.path.dirname(RetrievalConfig.PATH_PARSED_QUERIES), exist_ok=True)
            with open(RetrievalConfig.PATH_PARSED_QUERIES, 'w', encoding='utf-8') as f:
                json.dump(self.parsed_queries_cache, f, ensure_ascii=False, indent=2)

        # -----------------------------------------------------------------
        # 5. 打印最终结果表格
        # -----------------------------------------------------------------
        print("\n" + "="*85)
        print(f"{'Metric':<10} | {'Base':<8} | {'Ours(Score)':<12} | {'Ours(RRF)':<12} | {'Lift(Score)':<12} | {'Lift(RRF)':<12}")
        print("-" * 85)
        
        for k in RetrievalConfig.RECALL_KS:
            base_avg = np.mean(results_summary["Base"][k])
            score_avg = np.mean(results_summary["Ours_Score"][k])
            rrf_avg = np.mean(results_summary["Ours_RRF"][k])
            
            lift_score = (score_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0.0
            lift_rrf = (rrf_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0.0
            
            print(f"Recall@{k:<4} | {base_avg:.4f}   | {score_avg:.4f}       | {rrf_avg:.4f}       | {lift_score:+.2f}%       | {lift_rrf:+.2f}%")
        print("="*85)

if __name__ == "__main__":
    exp = CascadeRetrievalExperiment()
    exp.run()