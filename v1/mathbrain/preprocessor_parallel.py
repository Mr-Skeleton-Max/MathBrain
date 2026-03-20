#!/usr/bin/env python3
"""并行预处理模块

使用多进程并行处理大规模语料
"""

import numpy as np
import time
import pickle
from typing import List
from multiprocessing import Pool, cpu_count
from functools import partial

from .preprocessor_fast import FastGlobalPreprocessor, SentencePreprocessed


def _preprocess_chunk(sentences_chunk, model_bytes, chunk_id):
    """预处理一个句子块（worker 函数）

    Args:
        sentences_chunk: 句子列表
        model_bytes: 序列化的模型
        chunk_id: 块 ID

    Returns:
        (chunk_id, preprocessed_sentences)
    """
    # 反序列化模型
    model = pickle.loads(model_bytes)

    # 创建预处理器
    preprocessor = FastGlobalPreprocessor(model)

    # 预处理这个块
    preprocessed = []
    for sentence in sentences_chunk:
        sent_data = preprocessor._preprocess_sentence_fast(sentence)
        if sent_data is not None:
            preprocessed.append(sent_data)

    return (chunk_id, preprocessed)


class ParallelGlobalPreprocessor:
    """并行全局预处理器

    使用多进程并行处理大规模语料
    """

    def __init__(self, model, n_workers=None, chunk_size=1000):
        """初始化

        Args:
            model: MathBrain 模型
            n_workers: 并行工作进程数（默认为 CPU 核心数）
            chunk_size: 每个块的最小句子数（避免块太小导致开销过大）
        """
        self.model = model
        self.n_workers = n_workers or cpu_count()
        self.chunk_size = chunk_size

    def preprocess_corpus(self, sentences: List[str], verbose=True) -> List[SentencePreprocessed]:
        """并行预处理整个语料

        Args:
            sentences: 原始句子列表
            verbose: 是否显示进度

        Returns:
            预处理后的句子列表
        """
        # 如果语料太小，直接使用串行
        if len(sentences) < self.chunk_size * 2:
            if verbose:
                print(f"语料规模较小 ({len(sentences)} 句)，使用串行预处理")
            preprocessor = FastGlobalPreprocessor(self.model)
            return preprocessor.preprocess_corpus(sentences, verbose=verbose)

        if verbose:
            print(f"并行预处理: {len(sentences)} 句 ({self.n_workers} workers)")

        t0 = time.time()

        # 序列化模型（只序列化一次）
        if verbose:
            print("  序列化模型...")
        t_serial = time.time()
        model_bytes = pickle.dumps(self.model)
        if verbose:
            print(f"    耗时: {(time.time() - t_serial)*1000:.2f}ms")

        # 将语料分成合适大小的块
        actual_chunk_size = max(self.chunk_size, len(sentences) // self.n_workers)
        chunks = []
        for i in range(0, len(sentences), actual_chunk_size):
            chunk = sentences[i:i + actual_chunk_size]
            chunks.append((chunk, i // actual_chunk_size))

        if verbose:
            print(f"  分成 {len(chunks)} 块，每块约 {actual_chunk_size} 句")

        # 并行处理
        if verbose:
            print("  并行处理...")
        t_process = time.time()

        with Pool(processes=min(self.n_workers, len(chunks))) as pool:
            # 使用 starmap 传递多个参数
            results = pool.starmap(
                _preprocess_chunk,
                [(chunk, model_bytes, chunk_id) for chunk, chunk_id in chunks]
            )

        if verbose:
            print(f"    耗时: {(time.time() - t_process)*1000:.2f}ms")

        # 按原始顺序合并结果
        results.sort(key=lambda x: x[0])  # 按 chunk_id 排序
        preprocessed = []
        for _, chunk_result in results:
            preprocessed.extend(chunk_result)

        elapsed = (time.time() - t0) * 1000

        if verbose:
            print(f"  完成: {elapsed:.2f}ms ({elapsed/len(preprocessed):.4f}ms/句)")
            print(f"  实际吞吐: {len(preprocessed) / (elapsed/1000):.0f} 句/秒")

        return preprocessed


class HybridPreprocessor:
    """混合预处理器

    根据语料规模自动选择串行或并行
    """

    def __init__(self, model, n_workers=None, parallel_threshold=1000):
        """初始化

        Args:
            model: MathBrain 模型
            n_workers: 并行工作进程数
            parallel_threshold: 并行处理的最小句子数阈值
        """
        self.model = model
        self.n_workers = n_workers or cpu_count()
        self.parallel_threshold = parallel_threshold

        self.serial_preprocessor = FastGlobalPreprocessor(model)
        self.parallel_preprocessor = ParallelGlobalPreprocessor(model, n_workers)

    def preprocess_corpus(self, sentences: List[str], verbose=True) -> List[SentencePreprocessed]:
        """预处理整个语料（自动选择串行或并行）

        Args:
            sentences: 原始句子列表
            verbose: 是否显示进度

        Returns:
            预处理后的句子列表
        """
        if len(sentences) < self.parallel_threshold:
            # 小规模语料：使用串行
            if verbose:
                print(f"语料规模较小 ({len(sentences)} < {self.parallel_threshold})，使用串行预处理")
            return self.serial_preprocessor.preprocess_corpus(sentences, verbose=verbose)
        else:
            # 大规模语料：使用并行
            return self.parallel_preprocessor.preprocess_corpus(sentences, verbose=verbose)
