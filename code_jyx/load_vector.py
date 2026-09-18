# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# import math
# import torch
# import torch.multiprocessing as mp
# from datasets import load_dataset
# from pymilvus import MilvusClient
# from FlagEmbedding import BGEM3FlagModel
# from tqdm import tqdm
# import numpy as np
# import warnings

# # ================= 0. 屏蔽烦人的警告 =================
# warnings.filterwarnings("ignore", category=UserWarning)

# # ================= 配置区域 =================
# # 获取当前脚本所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 假设模型就在当前目录下 (如果不在，请修改绝对路径)
# EMBEDDING_MODEL_PATH = os.path.join(current_dir, 'bge-m3')

# # 临时文件夹，用于存放每个 GPU 计算出的片段
# TEMP_DIR = os.path.join(current_dir, "temp_embeddings")
# if not os.path.exists(TEMP_DIR):
#     os.makedirs(TEMP_DIR)

# datasets_config = [
#     {"name": "CovidRetrieval", "hf_path": "C-MTEB/CovidRetrieval", "split": "corpus", "doc_column": "text", "id_column": "id"},
#     {"name": "CmedqaRetrieval", "hf_path": "C-MTEB/CmedqaRetrieval", "split": "corpus", "doc_column": "text", "id_column": "id"},
#     {"name": "EcomRetrieval", "hf_path": "C-MTEB/EcomRetrieval", "split": "corpus", "doc_column": "text", "id_column": "id"}
# ]

# # ================= 核心：GPU 工作函数 =================

# def _worker_process(rank, gpu_id, texts, model_path, output_file):
#     """
#     独立进程：加载模型 -> 计算 -> 保存为 .npy 文件
#     """
#     try:
#         device = f"cuda:{gpu_id}"
#         print(f"[GPU {gpu_id}] 正在加载模型... (处理 {len(texts)} 条数据)")
        
#         # 加载模型
#         model = BGEM3FlagModel(model_path, use_fp16=True, device=device)
        
#         # 开始编码
#         # batch_size 可以根据 V100 显存适当调大，比如 32 或 64
#         embeddings = model.encode(texts, batch_size=64, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
        
#         # 保存结果到临时文件 (使用 numpy 格式，极快且省内存)
#         np.save(output_file, embeddings)
#         print(f"[GPU {gpu_id}] 计算完成，结果已保存至 {output_file}")
        
#     except Exception as e:
#         print(f"!!! [GPU {gpu_id}] 发生严重错误: {e}")
#         # 发生错误时保存一个空文件标记，避免主进程死等
#         np.save(output_file, np.array([]))

# def multi_gpu_encode_stable(all_texts, model_path):
#     """
#     主控函数：手动管理 Process，不使用 Pool
#     """
#     num_gpus = torch.cuda.device_count()
#     total_docs = len(all_texts)
    
#     # 1. 切分数据
#     chunk_size = math.ceil(total_docs / num_gpus)
#     processes = []
#     temp_files = []
    
#     print(f"--- 启动 8 卡并行计算 (总数据量: {total_docs}) ---")
    
#     # 2. 启动进程
#     for i in range(num_gpus):
#         start = i * chunk_size
#         end = min((i + 1) * chunk_size, total_docs)
        
#         if start >= end:
#             break
            
#         chunk_texts = all_texts[start:end]
#         output_file = os.path.join(TEMP_DIR, f"part_{i}.npy")
#         temp_files.append(output_file)
        
#         # 创建进程 (Process 默认不是守护进程，允许进行复杂的 CUDA 操作)
#         p = mp.Process(
#             target=_worker_process,
#             args=(i, i, chunk_texts, model_path, output_file)
#         )
#         p.start()
#         processes.append(p)

#     # 3. 等待所有进程结束
#     for p in processes:
#         p.join()

#     # 4. 合并结果
#     print("--- 所有 GPU 计算完毕，正在合并结果 ---")
#     all_embeddings_list = []
    
#     for f in temp_files:
#         if os.path.exists(f):
#             data = np.load(f)
#             if data.size > 0:
#                 all_embeddings_list.append(data)
#             # 删除临时文件
#             os.remove(f)
#         else:
#             print(f"警告：找不到文件 {f}")

#     if not all_embeddings_list:
#         return np.array([])

#     final_embeddings = np.concatenate(all_embeddings_list, axis=0)

#     # 强制转换为 float32，以满足 Milvus 的要求
#     print(f"正在将数据从 {final_embeddings.dtype} 转换为 float32...")
#     final_embeddings = final_embeddings.astype(np.float32)

#     print(f"合并完成，最终向量形状: {final_embeddings.shape}")
#     return final_embeddings

# # ================= 主流程 =================

# def process_and_insert(config):
#     client = MilvusClient("experiment_data.db")
#     collection_name = config["name"]
    
#     # 1. 重建集合
#     if client.has_collection(collection_name):
#         client.drop_collection(collection_name)
    
#     client.create_collection(
#         collection_name=collection_name,
#         dimension=1024,
#         auto_id=False,
#         primary_field_name="id",
#         vector_field_name="vector",
#         id_type="string",  # 显式指定主键为字符串
#         max_length=128,    # 字符串主键必须指定最大长度 (128足够了)
#         description=f"Corpus for {collection_name}"
#     )
#     print(f"====== 集合 {collection_name} 创建成功 ======")

#     # 2. 加载数据
#     print(f"正在加载数据 {config['hf_path']}...")
#     try:
#         dataset = load_dataset(config["hf_path"], split=config["split"])
#         # dataset = dataset.select(range(1000)) # 调试用
        
#         all_texts = dataset[config["doc_column"]]
#         all_ids = dataset[config["id_column"]]
#         print(f"数据加载完成，共 {len(all_texts)} 条。")
#     except Exception as e:
#         print(f"数据加载失败: {e}")
#         return

#     # 3. 执行稳定的多卡并行
#     all_vectors = multi_gpu_encode_stable(all_texts, EMBEDDING_MODEL_PATH)

#     # 4. 写入 Milvus
#     print(f"正在写入 Milvus...")
#     batch_size = 1000 
    
#     # 确保向量数量和ID数量一致
#     if len(all_vectors) != len(all_ids):
#         print(f"严重警告：向量数量 ({len(all_vectors)}) 与 ID 数量 ({len(all_ids)}) 不一致！")
#         return

#     for i in tqdm(range(0, len(all_texts), batch_size), desc="入库进度"):
#         batch_end = i + batch_size
        
#         batch_ids = all_ids[i : batch_end]
#         batch_texts = all_texts[i : batch_end]
#         batch_vecs = all_vectors[i : batch_end]
        
#         insert_data = []
#         for j in range(len(batch_ids)):
#             insert_data.append({
#                 "id": str(batch_ids[j]),
#                 # 显式转换为列表，这是最兼容 Milvus 的方式
#                 "vector": batch_vecs[j].tolist(), 
#                 "text": batch_texts[j][0:1000] # 截断
#             })
            
#         client.insert(collection_name=collection_name, data=insert_data)
        
#     print(f"任务 {collection_name} 完成！\n")

# if __name__ == "__main__":
#     # 强制设置启动方式为 spawn (解决 CUDA 多进程的关键)
#     mp.set_start_method('spawn', force=True)

#     for config in datasets_config:
#         process_and_insert(config)

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import math
import json
import torch
import torch.multiprocessing as mp
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import numpy as np
import warnings

# ================= 0. 屏蔽烦人的警告 =================
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 配置区域 =================
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设模型就在当前目录下 (如果不在，请修改绝对路径)
EMBEDDING_MODEL_PATH = os.path.join(current_dir, 'bge-m3')

# 临时文件夹，用于存放每个 GPU 计算出的片段
TEMP_DIR = os.path.join(current_dir, "temp_embeddings")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# 修改为 LexRAG 本地数据配置
datasets_config =[
    {
        "name": "LexRAG_LegalArticles", 
        "local_path": "LexRAG/data/law_library.jsonl" # 确保该路径正确
    }
]

# ================= 核心：GPU 工作函数 =================

def _worker_process(rank, gpu_id, texts, model_path, output_file):
    """
    独立进程：加载模型 -> 计算 -> 保存为 .npy 文件
    """
    try:
        device = f"cuda:{gpu_id}"
        print(f"[GPU {gpu_id}] 正在加载模型... (处理 {len(texts)} 条数据)")
        
        # 加载模型
        model = BGEM3FlagModel(model_path, use_fp16=True, device=device)
        
        # 开始编码
        # 注意: 如果法条文本很长导致 OOM，可以适当调小 batch_size (如 32 或 16)
        embeddings = model.encode(texts, batch_size=32, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
        
        # 保存结果到临时文件
        np.save(output_file, embeddings)
        print(f"[GPU {gpu_id}] 计算完成，结果已保存至 {output_file}")
        
    except Exception as e:
        print(f"!!! [GPU {gpu_id}] 发生严重错误: {e}")
        # 发生错误时保存一个空文件标记，避免主进程死等
        np.save(output_file, np.array([]))

def multi_gpu_encode_stable(all_texts, model_path):
    """
    主控函数：手动管理 Process，不使用 Pool
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("未检测到可用的 GPU！")
        
    total_docs = len(all_texts)
    
    # 1. 切分数据
    chunk_size = math.ceil(total_docs / num_gpus)
    processes = []
    temp_files =[]
    
    print(f"--- 启动 {num_gpus} 卡并行计算 (总数据量: {total_docs}) ---")
    
    # 2. 启动进程
    for i in range(num_gpus):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_docs)
        
        if start >= end:
            break
            
        chunk_texts = all_texts[start:end]
        output_file = os.path.join(TEMP_DIR, f"part_{i}.npy")
        temp_files.append(output_file)
        
        p = mp.Process(
            target=_worker_process,
            args=(i, i, chunk_texts, model_path, output_file)
        )
        p.start()
        processes.append(p)

    # 3. 等待所有进程结束
    for p in processes:
        p.join()

    # 4. 合并结果
    print("--- 所有 GPU 计算完毕，正在合并结果 ---")
    all_embeddings_list =[]
    
    for f in temp_files:
        if os.path.exists(f):
            data = np.load(f)
            if data.size > 0:
                all_embeddings_list.append(data)
            os.remove(f)
        else:
            print(f"警告：找不到文件 {f}")

    if not all_embeddings_list:
        return np.array([])

    final_embeddings = np.concatenate(all_embeddings_list, axis=0)

    # 强制转换为 float32，以满足 Milvus 的要求
    final_embeddings = final_embeddings.astype(np.float32)

    print(f"合并完成，最终向量形状: {final_embeddings.shape}")
    return final_embeddings

# ================= 本地数据读取工具 =================
def load_lexrag_corpus(file_path):
    """读取本地 JSONL 格式的 LexRAG 法条语料库"""
    all_texts =[]
    all_ids =[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            
            # 提取法条 ID (例如："《中华人民共和国民法典》第四百二十八条")
            doc_id = str(item.get("id", item.get("doc_id", f"law_{idx}")))
            all_ids.append(doc_id)
            
            # 提取法条文本
            if "text" in item:
                text_content = item["text"]
            elif "content" in item:
                text_content = item["content"]
            else:
                text_content = json.dumps(item, ensure_ascii=False)
            all_texts.append(text_content)
            
    return all_ids, all_texts


# ================= 主流程 =================

def process_and_insert(config):
    # 建立一个专属的向量数据库，避免污染之前的实验
    client = MilvusClient("lexrag_vector.db") 
    collection_name = config["name"]
    
    # 1. 重建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=1024, # BGE-M3 的 dense 向量维度是 1024
        auto_id=False,
        primary_field_name="id",
        vector_field_name="vector",
        id_type="string",  
        max_length=512,    # 【关键修改】法条名称可能很长，128可能不够，调大到 512
        description=f"Corpus for {collection_name}"
    )
    print(f"====== 集合 {collection_name} 创建成功 ======")

    # 2. 加载本地数据
    print(f"正在加载本地数据 {config['local_path']}...")
    if not os.path.exists(config['local_path']):
        print(f"❌ 错误：未找到文件 {config['local_path']}。请检查路径。")
        return
        
    try:
        all_ids, all_texts = load_lexrag_corpus(config["local_path"])
        print(f"数据加载完成，共 {len(all_texts)} 条。")
        # print("示例 ID:", all_ids[0])
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 3. 执行稳定的多卡并行
    all_vectors = multi_gpu_encode_stable(all_texts, EMBEDDING_MODEL_PATH)

    # 4. 写入 Milvus
    print(f"正在写入 Milvus...")
    batch_size = 1000 
    
    # 确保向量数量和ID数量一致
    if len(all_vectors) != len(all_ids):
        print(f"严重警告：向量数量 ({len(all_vectors)}) 与 ID 数量 ({len(all_ids)}) 不一致！")
        return

    for i in tqdm(range(0, len(all_texts), batch_size), desc="入库进度"):
        batch_end = i + batch_size
        
        batch_ids = all_ids[i : batch_end]
        batch_texts = all_texts[i : batch_end]
        batch_vecs = all_vectors[i : batch_end]
        
        insert_data =[]
        for j in range(len(batch_ids)):
            insert_data.append({
                "id": str(batch_ids[j]),
                "vector": batch_vecs[j].tolist(), 
                "text": batch_texts[j][0:2000] # 法条比较长，如果需要存截断信息可以调大
            })
            
        client.insert(collection_name=collection_name, data=insert_data)
        
    print(f"任务 {collection_name} 完成！\n")

if __name__ == "__main__":
    # 强制设置启动方式为 spawn (解决 CUDA 多进程的关键)
    mp.set_start_method('spawn', force=True)

    for config in datasets_config:
        process_and_insert(config)