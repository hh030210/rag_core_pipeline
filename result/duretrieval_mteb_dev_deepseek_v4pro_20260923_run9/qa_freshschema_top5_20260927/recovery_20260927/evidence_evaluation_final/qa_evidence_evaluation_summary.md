# DuRetrieval 证据型问答评测（重试后）

该数据集没有人工参考答案；分数依据 qrels 相关证据，由 LLM 评测，不等同于人工答案准确率。修复了可重试的空答案，并对原失败评测项及修复答案进行了复评。

- correctness: 1.740122
- completeness: 1.559017
- relevance: 1.832827
- groundedness: 1.719858
- unsupported_claims: 0.277862
- evidence_overlap: 0.812914
- 评测失败数: 26
- 有效评分数: 1974/2000
