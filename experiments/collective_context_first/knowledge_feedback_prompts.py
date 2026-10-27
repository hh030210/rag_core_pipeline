"""Qualitative feedback aligned to context-first/internal-knowledge answering."""
from collective_feedback_prompts import DIAGNOSIS_SPEC as ORIGINAL_DIAG, UPDATE_SPEC as ORIGINAL_UPDATE
from knowledge_fallback_workflow import EVIDENCE,GOALS

DIAGNOSIS_SPEC=ORIGINAL_DIAG.replace(
 'actionable为true时，须同时给出两份答案及实际上下文的非空引文，说明对应提示规则的问题及可迁移的修复方法。',
 'actionable为true时，须给出两份答案的非空引文及可迁移修复原则。涉及材料已有内容时给实际上下文的逐字依据；纯知识补全部分允许context_excerpt为空，不伪造材料引文。')
DIAGNOSIS_SPEC+='''
[本轮判断边界]
先核对上下文已有信息的使用，再核对缺失部分的模型知识补全。supported包括材料支持或自身知识可合理核实，不只指材料逐字出现；没有材料依据不自动等于unsupported。内部知识无法核实则uncertain，不预设案例答案或参考答案为真。
事实正确性不使用参考答案；问题覆盖和需求契合只依据问题与两份实际答案；参考答案仅辅助描述语义，不新增必答项。补充内容不在上下文忠实度范围内，但不得冒称材料原文。
不使用数字评分或固定维度总分驱动本环节。以下为作答边界，不要求输出这些指标的评分：
'''+GOALS

start=ORIGINAL_UPDATE.index('保留原有问答结构、作答质量要求及其具体操作方法。')
end=ORIGINAL_UPDATE.index('完整回答必要问项后尽可能简洁。',start)
UPDATE_SPEC=ORIGINAL_UPDATE[:start]+'保留原有问答结构和具体操作方法。\n'+EVIDENCE+'\n'+GOALS+'\n'+ORIGINAL_UPDATE[end:]
DIAGNOSIS_META='''[角色]
你是群智提示融合损失的诊断指令设计专家。
[任务]
生成一份直接对比案例级与群智答案的通用诊断指令，不输入具体案例，不再生成指令。
[成品输入]
question、reference_answer、contexts、case_optimized（prompt、answer）、current_collective（prompt、answer）、previous_collective（可空）、answer_policy。
[必须保留的诊断规范]
'''+DIAGNOSIS_SPEC+'''
[输出]
只输出成品诊断指令，使用[角色][任务][输入]，保留全部定性字段、连续逐字引文、旧评分不可见和数据边界。允许未发现任何问题，不增加数字评分。'''
UPDATE_META='''[角色]
你是群智QA提示反馈更新指令设计专家。
[任务]
生成一份直接改写当前簇QA提示的通用更新指令，不输入具体案例，不再生成更新指令。
[成品输入]
cluster_id、current_collective_prompt、training_cases（问题、参考、实际上下文、案例级提示和答案、群智答案、质性诊断）。
[必须保留的更新规范]
'''+UPDATE_SPEC+'''
[输出]
只输出成品更新指令，使用[角色][任务][输入]；执行更新时只输出六段结构的QA正文。保留本轮先上下文、缺失时允许内部知识补全的边界，不依据数字评分更新。'''
