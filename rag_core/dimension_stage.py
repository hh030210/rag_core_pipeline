"""维度阶段适配层：生产逻辑来自原始 ``code_jyx``。

新项目不重新实现维度 Schema、标签抽取和倒排索引。这里仅负责把三阶段
分片产生的 chunk 记录转换成原始 code_jyx v2 接口需要的输入，并保存到
当前 run-dir。叶子维度、多标签、evidence、父子层次和 v2 索引语义均由
``code_jyx`` 中的原实现负责。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from code_jyx.dimension_generate import V2_THRESHOLDS, diagnose_leaf_dimensions
from code_jyx.index_builder import IndexBuilderV2
from code_jyx.schema_v2 import SCHEMA_VERSION, load_schema
from code_jyx.tag_generate import TagGeneratorV2
from code_jyx.v2_pipeline import MockMiner, make_mock_schema


def default_schema() -> Dict[str, Any]:
    """返回原 code_jyx v2 测试 Schema，保留两层叶子维度语义。"""
    return make_mock_schema()


def build_tags(records: Iterable[Dict[str, Any]], schema: Dict[str, Any], settings,
               miner=None) -> Dict[str, Any]:
    """调用原始 ``TagGeneratorV2`` 完成叶子多标签抽取。"""
    if settings.mock:
        miner = MockMiner()
    if miner is None:
        raise ValueError("真实维度抽取需要 code_jyx.DimensionMiningWithQwen")
    generator = TagGeneratorV2(
        schema,
        miner=miner,
        max_dimensions_per_request=settings.max_dimensions_per_request,
    )
    return generator.run(list(records))


def build_inverted_index(tag_output: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """调用原始 ``IndexBuilderV2``，不再使用新项目的重复实现。"""
    return IndexBuilderV2(schema).build(tag_output)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dimensions(records, settings, schema_path: str = "") -> Dict[str, Any]:
    """运行原始 code_jyx v2 Schema、叶子标签和倒排索引流程。"""
    settings.ensure_dirs()
    records = list(records or [])
    if not records:
        raise ValueError("维度抽取输入 records 为空")

    if schema_path:
        schema = load_schema(schema_path, allow_legacy=True, strict=False)
        generation_method = "provided_schema"
    elif settings.mock or not settings.llm_api_key:
        schema = default_schema()
        generation_method = "code_jyx_mock_schema"
    else:
        from code_jyx.llm_service import DimensionMiningWithQwen

        miner_for_schema = DimensionMiningWithQwen(
            api_key=settings.llm_api_key,
            model_name=settings.llm_model,
            base_url=settings.llm_base_url,
        )
        schema = miner_for_schema.generate_candidate_schema(
            [str(item.get("doc_text", item.get("chunk_text_full", ""))) for item in records]
        )
        schema = load_schema(schema, allow_legacy=False, strict=True)
        generation_method = "code_jyx_llm_v2"

    schema = {**schema, "generation_method": generation_method}
    _write_json(settings.run_dir / "V_cand_v2.json", schema)
    _write_json(settings.run_dir / "V_core_v2.json", schema)

    miner = None
    if not settings.mock:
        from code_jyx.llm_service import DimensionMiningWithQwen

        miner = DimensionMiningWithQwen(
            api_key=settings.llm_api_key,
            model_name=settings.llm_model,
            base_url=settings.llm_base_url,
        )
    tag_output = build_tags(records, schema, settings, miner=miner)
    _write_json(settings.run_dir / "tags_output_v2.json", tag_output)

    index = build_inverted_index(tag_output, schema)
    builder = IndexBuilderV2(schema, output_dir=settings.run_dir)
    builder.save(index, settings.run_dir)

    # 原实现提供的叶子覆盖率、最小支持数和兄弟标签重叠诊断也保留。
    diagnostics = diagnose_leaf_dimensions(
        tag_output, schema, thresholds=getattr(settings, "dimension_thresholds", V2_THRESHOLDS)
    )
    _write_json(settings.run_dir / "dimension_diagnostics_v2.json", diagnostics)

    return {"schema": schema, "tags": tag_output, "index": index,
            "diagnostics": diagnostics}
