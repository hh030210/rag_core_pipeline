# DuRetrieval 证据型问答评测（最终复核）

该数据集没有人工参考答案；分数基于 qrels 相关证据的 LLM 判定，不等同于人工答案准确率。空答案已作有界重试；`evidence_overlap` 按最终答案和 gold evidence 重新计算。

- correctness: 1.742424
- completeness: 1.559848
- relevance: 1.833838
- groundedness: 1.722727
- unsupported_claims: 0.27702
- evidence_overlap: 0.815378
- LLM 评测失败数: 20
- 有效 LLM 判定: 1980/2000
- 仍为空的生成答案: 33/2000
