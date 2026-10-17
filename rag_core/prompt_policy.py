"""Five-module schema and immutable evidence-use requirements."""
from pathlib import Path
import re

MODULE_FIELDS = (
    "system_prompt", "instruction", "context_strategy",
    "format_requirement", "uncertainty_handling",
)
NOISE_REQUIREMENT = "给定上下文可能包含噪音、无关信息或相互矛盾的信息。"
RERANK_REQUIREMENT = (
    "无论给定多少条上下文，都必须先根据当前问题对全部上下文进行相关性和证据价值重排序（rerank），"
    "仅保留并使用你判断对回答有用的上下文，丢弃无用或不可靠的信息；不设固定保留条数，"
    "允许全部保留或全部舍弃，证据不足时明确说明。"
)
POLICY_TEXT = NOISE_REQUIREMENT + "\n" + RERANK_REQUIREMENT
META_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "case_initial_meta.txt"


WITHOUT_POLICY_GUIDANCE = (
    "本组只生成普通的五模块知识库问答提示。context_strategy 可说明基于给定上下文核验实体、属性、关系并综合作答。"
    "不要在生成的五模块中增加关于上下文含噪音或噪声的声明，也不要增加对上下文排序、重排序、"
    "rerank、筛选、过滤或只保留有用片段的操作要求。保留证据约束、事实准确、完整作答与不确定性处理。"
)
WITHOUT_POLICY_PATTERN = re.compile(
    r"噪[音声]|去噪|rerank|re[- ]?rank|重排|重新排序|重排序|筛选|过滤|剔除|舍弃|丢弃|"
    r"(?:只|仅)保留|保留[^。；\n]{0,30}(?:有用|相关)(?:的)?(?:上下文|片段|证据)", re.I)


def validate_module(raw, evidence_policy=True):
    if not isinstance(raw, dict):
        raise ValueError("Prompt must be a JSON object")
    for key in MODULE_FIELDS:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"Missing non-empty prompt field: {key}")
    module = {key: raw[key].strip() for key in MODULE_FIELDS}
    # Exact immutable clauses, not a heuristic keyword/semantic check.
    if evidence_policy and not all(clause in module["context_strategy"] for clause in (NOISE_REQUIREMENT, RERANK_REQUIREMENT)):
        raise ValueError("context_strategy must contain both immutable evidence policy clauses verbatim")
    if not evidence_policy and WITHOUT_POLICY_PATTERN.search("\n".join(module.values())):
        raise ValueError("Control prompt contains excluded noise/rerank/filter instructions")
    return module


def protect_module(raw, evidence_policy=True):
    """Protect legacy/fallback modules at inference, without accepting bad candidates."""
    module = {key: str(raw.get(key, "")) for key in MODULE_FIELDS}
    if evidence_policy:
        for clause in (NOISE_REQUIREMENT, RERANK_REQUIREMENT):
            if clause not in module["context_strategy"]:
                module["context_strategy"] += "\n" + clause
    return module


def request_module(client, system, user, *, max_tokens=2200, evidence_policy=True):
    """Reject invalid generations; one explicit repair attempt, never a silent default."""
    last_error = None
    for attempt in range(2):
        retry = "" if not attempt else f"\n上次输出无效：{last_error}。请重新生成，严格满足全部字段和不可删除约束。"
        try:
            policy_guidance = ("\n不可删除、不可改写的上下文策略原文：\n" + POLICY_TEXT
                               if evidence_policy else "\n" + WITHOUT_POLICY_GUIDANCE)
            return validate_module(client.complete_json(
                system, user + retry + policy_guidance,
                temperature=0.1, max_tokens=max_tokens,
                repair_prompt=("仅修复 JSON 语法，保留原提示的事实约束和五个非空字符串字段："
                    + "、".join(MODULE_FIELDS) + "。不要添加其他字段。" + policy_guidance),
            ), evidence_policy=evidence_policy)
        except (ValueError, TypeError, KeyError) as exc:
            last_error = exc
    raise ValueError(f"Prompt generation failed policy/schema validation: {last_error}")
