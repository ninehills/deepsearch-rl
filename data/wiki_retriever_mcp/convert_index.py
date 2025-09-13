import numpy as np
import faiss
import gc
import os
import time
import argparse

# ==============================================================================
# 0. 全局配置
# ==============================================================================
INDEX_TYPE = "IVFPQ"      # "IVFPQ" 或 "HNSWPQ"
CONFIG = {
    # --- 文件路径 ---
    "flat_index_path": "_data/e5_Flat.index",
    "final_index_path": f"_data/e5_{INDEX_TYPE.lower()}.faiss",

    # --- 索引类型选择 ---
    "index_type": INDEX_TYPE,
    
    # --- 索引参数 ---
    "d": 768,         # 向量维度
    "nlist": 1024,    # IVF聚类中心数 (仅IVFPQ使用)
    "m": 96,          # PQ子向量数
    "nbits": 8,       # 每个子向量量化比特数
    
    # --- HNSW特有参数 ---
    "M": 32,                    # Maximum number of connections per layer
    "efConstruction": 200,      # HNSW构建时ef值 (200-500推荐)
    "efSearch": 64,             # HNSW搜索时ef值 (32-128)

    # --- 流程控制参数 ---
    "n_train_vectors": 1_000_000,  # 用于训练的向量数
    "add_batch_size": 200_000,     # 分批添加数据的批量大小
    
    # --- 基准测试参数 ---
    "benchmark_nq": 100,         # 基准测试的查询向量数 (100个以获得较快反馈)
    "benchmark_k": 5,           # 查找Top K
    
    # --- 随机种子 ---
    "random_seed": 42,          # 全局随机种子
}

# 设置随机种子
np.random.seed(CONFIG["random_seed"])

# ==============================================================================
# 1. 功能函数：构建量化索引
# ==============================================================================
def build_quantized_index():
    """
    从 flat.index 构建、训练并填充量化索引。
    训练过程会使用GPU加速，最终索引保存在CPU端。
    """
    print("--- 开始构建量化索引 ---")
    
    # --- 安全检查 ---
    if not os.path.exists(CONFIG["flat_index_path"]):
        print(f"错误: 原始索引 '{CONFIG['flat_index_path']}' 不存在。请先准备好该文件。")
        return

    if os.path.exists(CONFIG["final_index_path"]):
        overwrite = input(f"警告: 目标索引 '{CONFIG['final_index_path']}' 已存在。是否覆盖? (y/n): ").lower()
        if overwrite != 'y':
            print("操作已取消。")
            return
    
    total_start_time = time.time()

    # --------------------------------------------------------------------------
    # 步骤 1.1: 准备训练数据 (内存管理是关键)
    # --------------------------------------------------------------------------
    print(f"\n[1/4] 准备训练数据...")
    
    # 加载巨大的flat index
    print(f"    - 加载 '{CONFIG['flat_index_path']}'...")
    index_flat = faiss.read_index(CONFIG["flat_index_path"])
    ntotal = index_flat.ntotal
    assert index_flat.d == CONFIG["d"], f"索引维度({index_flat.d})与配置维度({CONFIG['d']})不符！"

    # 直接按随机索引重建向量，避免重建全部
    print(f"    - 从 {ntotal} 中采样 {CONFIG['n_train_vectors']} 个向量...")
    random_indices = np.random.choice(ntotal, CONFIG["n_train_vectors"], replace=False)
    
    # 使用reconstruct_n逐个获取向量
    xt_list = []
    for idx in random_indices:
        vec = index_flat.reconstruct_n(int(idx), 1)
        xt_list.append(vec)
    
    xt = np.vstack(xt_list)
    xt = np.ascontiguousarray(xt, dtype='float32')

    # 立即释放内存
    print("    - 释放原始索引以回收内存...")
    del index_flat
    gc.collect()
    print("    - 训练数据准备完毕。")

    # --------------------------------------------------------------------------
    # 步骤 1.2: 创建并训练索引
    # --------------------------------------------------------------------------
    print(f"\n[2/4] 创建并训练{CONFIG['index_type']}索引...")
    train_start_time = time.time()
    
    if CONFIG["index_type"] == "IVFPQ":
        # 创建IVFPQ索引
        quantizer = faiss.IndexFlatL2(CONFIG["d"])
        index = faiss.IndexIVFPQ(quantizer, CONFIG["d"], CONFIG["nlist"], CONFIG["m"], CONFIG["nbits"])
        
        try:
            print("    - 尝试在GPU上训练IVFPQ索引...")
            res = faiss.StandardGpuResources()
            config = faiss.GpuClonerOptions()
            config.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index, config)
            gpu_index.train(xt)
            index = faiss.index_gpu_to_cpu(gpu_index)
            print("    - GPU训练完成。")
        except Exception as e:
            print(f"    - GPU训练失败: {e}")
            print("    - 切换到CPU进行训练...")
            index.train(xt)
            print("    - CPU训练完成。")
            
    elif CONFIG["index_type"] == "HNSWPQ":
        # 创建HNSWPQ索引
        print("    - 创建HNSWPQ索引...")
        # https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSWPQ.html
        index = faiss.IndexHNSWPQ(CONFIG["d"], CONFIG["m"], CONFIG["M"], CONFIG["nbits"], faiss.METRIC_L2)
        index.hnsw.efConstruction = CONFIG["efConstruction"]
        
        # HNSW训练只需要较少向量 (推荐至少39k，但我们用现有的训练集)
        print(f"    - 使用{len(xt)}个向量训练HNSWPQ...")
        index.train(xt)
        print("    - HNSWPQ训练完成。")
    
    else:
        raise ValueError(f"不支持的索引类型: {CONFIG['index_type']}")
    
    # 统一变量名
    index_final = index

    del xt
    gc.collect()
    print(f"    - 训练耗时: {time.time() - train_start_time:.2f} 秒。")

    # --------------------------------------------------------------------------
    # 步骤 1.3: 分批次添加所有向量（GPU加速）
    # --------------------------------------------------------------------------
    print(f"\n[3/4] 分批添加 {ntotal} 个向量...")
    add_start_time = time.time()
    
    index_flat = faiss.read_index(CONFIG["flat_index_path"]) # 再次加载
    num_batches = (ntotal + CONFIG["add_batch_size"] - 1) // CONFIG["add_batch_size"]
    
    # 根据索引类型选择添加策略
    if CONFIG["index_type"] == "IVFPQ":
        # IVFPQ可以尝试GPU加速
        try:
            print("    - 尝试使用GPU加速添加向量...")
            res = faiss.StandardGpuResources()
            config = faiss.GpuClonerOptions()
            config.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index_final, config)
            
            for i in range(num_batches):
                start_idx = i * CONFIG["add_batch_size"]
                end_idx = min((i + 1) * CONFIG["add_batch_size"], ntotal)
                
                print(f"\r    - GPU添加批次 {i+1}/{num_batches}...", end="")
                batch_vectors = index_flat.reconstruct_n(start_idx, end_idx - start_idx)
                gpu_index.add(batch_vectors)
            
            index_final = faiss.index_gpu_to_cpu(gpu_index)
            print(f"\n    - GPU添加完成。")
            
        except Exception as e:
            print(f"\n    - GPU添加失败: {e}")
            print("    - 切换到CPU添加向量...")
            
            for i in range(num_batches):
                start_idx = i * CONFIG["add_batch_size"]
                end_idx = min((i + 1) * CONFIG["add_batch_size"], ntotal)
                
                print(f"\r    - CPU添加批次 {i+1}/{num_batches}...", end="")
                batch_vectors = index_flat.reconstruct_n(start_idx, end_idx - start_idx)
                index_final.add(batch_vectors)
                
    else:
        # HNSWPQ只能在CPU上添加
        print("    - HNSWPQ使用CPU添加向量...")
        for i in range(num_batches):
            start_idx = i * CONFIG["add_batch_size"]
            end_idx = min((i + 1) * CONFIG["add_batch_size"], ntotal)
            
            print(f"\r    - CPU添加批次 {i+1}/{num_batches}...", end="")
            batch_vectors = index_flat.reconstruct_n(start_idx, end_idx - start_idx)
            index_final.add(batch_vectors)

    del index_flat
    gc.collect()
    print(f"\n    - 所有向量添加完毕。")
    print(f"    - 添加数据耗时: {time.time() - add_start_time:.2f} 秒。")
    assert index_final.ntotal == ntotal

    # --------------------------------------------------------------------------
    # 步骤 1.4: 保存最终索引
    # --------------------------------------------------------------------------
    print(f"\n[4/4] 保存最终索引...")
    faiss.write_index(index_final, CONFIG["final_index_path"])
    print(f"    - 索引成功保存到 '{CONFIG['final_index_path']}'")
    print(f"    - 索引大小: {os.path.getsize(CONFIG['final_index_path'])} 字节")
    
    print(f"\n--- 构建流程完成，总耗时: {time.time() - total_start_time:.2f} 秒 ---")

# ==============================================================================
# 2. 功能函数：运行CPU基准测试
# ==============================================================================
def run_cpu_benchmark():
    """
    在CPU上对原始索引和量化索引进行速度和精度测试。
    """
    print("--- 开始运行CPU基准测试 ---")

    # --- 安全检查 ---
    if not os.path.exists(CONFIG["flat_index_path"]) or not os.path.exists(CONFIG["final_index_path"]):
        print(f"错误: 必须同时存在 '{CONFIG['flat_index_path']}' 和 '{CONFIG['final_index_path']}' 才能进行测试。")
        print("如果尚未构建量化索引，请先运行 'build' 命令。")
        return

    # --------------------------------------------------------------------------
    # 步骤 2.1: 准备查询数据
    # --------------------------------------------------------------------------
    print(f"\n[1/3] 准备 {CONFIG['benchmark_nq']} 个查询向量...")
    index_flat = faiss.read_index(CONFIG["flat_index_path"])
    print(f"    - 索引加载成功。ntotal={index_flat.ntotal}, d={index_flat.d}")
    ntotal = index_flat.ntotal
    query_indices = np.random.choice(ntotal, CONFIG['benchmark_nq'], replace=False)
    
    # 使用reconstruct_n逐个获取查询向量（避免OOM）
    xq_list = []
    for idx in query_indices:
        vec = index_flat.reconstruct_n(int(idx), 1)
        xq_list.append(vec)
    
    xq = np.vstack(xq_list)
    xq = np.ascontiguousarray(xq, dtype='float32')
    print("    - 查询向量准备完毕。")

    # --------------------------------------------------------------------------
    # 步骤 2.2: 测试原始 IndexFlat (获取基准速度和真实结果)
    # --------------------------------------------------------------------------
    print(f"\n[2/3] 在 CPU 上测试 IndexFlat (获取基准)...")
    index_flat.search(xq[:5], CONFIG['benchmark_k']) 
    print(f"    - 预热完成，开始测试。")
    
    start_time = time.time()
    _, I_gt = index_flat.search(xq, CONFIG['benchmark_k']) # Ground Truth
    end_time = time.time()
    
    flat_time = end_time - start_time
    flat_qps = CONFIG['benchmark_nq'] / flat_time
    
    print(f"    - 搜索耗时: {flat_time * 1000:.2f} ms")
    print(f"    - 速度 (QPS): {flat_qps:.2f}")
    
    del index_flat
    gc.collect()

    # --------------------------------------------------------------------------
    # 步骤 2.3: 测试量化索引
    # --------------------------------------------------------------------------
    print(f"\n[3/3] 在 CPU 上测试 {CONFIG['index_type']}...")
    index_quantized = faiss.read_index(CONFIG["final_index_path"])
    
    if CONFIG["index_type"] == "IVFPQ":
        # IVFPQ测试不同nprobe值
        nprobes_to_test = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        test_params = nprobes_to_test
        param_name = "nprobe"
    else:
        # HNSWPQ测试不同efSearch值
        efsearch_to_test = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        test_params = efsearch_to_test  
        param_name = "efSearch"

    print("\n" + "-"*70)
    print(f"{param_name:<8} | Recall@{CONFIG['benchmark_k']:<2} | Search Time (ms) | QPS      | Speedup vs Flat")
    print("---------|----------|--------------------|----------|-----------------")
    
    for param_val in test_params:
        if CONFIG["index_type"] == "IVFPQ":
            index_quantized.nprobe = param_val
        else:
            index_quantized.hnsw.efSearch = param_val
        
        # 预热
        index_quantized.search(xq[:5], CONFIG['benchmark_k'])
        
        start_time = time.time()
        _, I_ann = index_quantized.search(xq, CONFIG['benchmark_k'])
        end_time = time.time()
        
        search_time = end_time - start_time
        search_qps = CONFIG['benchmark_nq'] / search_time
        
        # 计算精度
        correct_count = np.sum([len(set(I_gt[i]).intersection(set(I_ann[i]))) for i in range(CONFIG['benchmark_nq'])])
        recall = correct_count / (CONFIG['benchmark_nq'] * CONFIG['benchmark_k'])
        speedup = search_qps / flat_qps

        print(f"{param_val:<8} | {recall:<8.4f} | {search_time * 1000:<18.2f} | {search_qps:<8.2f} | {speedup:^15.2f}x")
    
    print("-" * 65)
    print("\n--- 基准测试完成 ---")

# ==============================================================================
# 3. 主函数：命令行入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="为大型Faiss Flat索引构建量化索引并在CPU上进行基准测试的工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help="可执行的命令")

    # 'build' 命令
    subparsers.add_parser('build', help='从 flat.index 构建并保存量化索引。')

    # 'benchmark' 命令
    subparsers.add_parser('benchmark', help='在CPU上比较 flat 和量化索引的性能。')

    args = parser.parse_args()

    if args.command == 'build':
        build_quantized_index()
    elif args.command == 'benchmark':
        run_cpu_benchmark()

if __name__ == "__main__":
    main()