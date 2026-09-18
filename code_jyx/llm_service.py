"""
llm_service.py

基于 DashScope (通义千问) 的 LLM 服务封装。
实现了 dimension_generate、tag_generate、query_parser 三个模块所需的所有 LLM 接口。

也支持 OpenAI 兼容协议（通过环境变量 LLM_OPENAI_COMPAT=1 启用），可用于阿里云 Maas /
SiliconFlow 等 OpenAI 风格 endpoint。

使用前请先设置环境变量：
    export DASHSCOPE_API_KEY="sk-..."
或在使用时直接传入 api_key 参数。

OpenAI 兼容模式：
    export LLM_OPENAI_COMPAT=1
    export LLM_BASE_URL="https://llm-xxx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    export LLM_API_KEY="sk-..."
    export LLM_MODEL="qwen3.6-max-preview"
"""

import os
import re
import json
import threading
import time
from typing import List, Dict, Optional, Any

try:
    from .schema_v2 import (
        SCHEMA_VERSION,
        SchemaValidationError,
        make_dimension_node,
        make_v2_schema,
        normalize_label,
        normalize_text,
        schema_maps,
        stable_dimension_id,
        is_stable_dimension_id,
        validate_schema,
    )
except ImportError:  # direct ``python code_jyx/llm_service.py`` compatibility
    from schema_v2 import (
        SCHEMA_VERSION,
        SchemaValidationError,
        make_dimension_node,
        make_v2_schema,
        normalize_label,
        normalize_text,
        schema_maps,
        stable_dimension_id,
        is_stable_dimension_id,
        validate_schema,
    )


# ============================================================
# OpenAI 兼容客户端（懒加载）
# ============================================================

_OPENAI_CLIENT_CACHE = {}


def _get_openai_client(base_url: str, api_key: str):
    """懒加载 OpenAI 客户端"""
    cache_key = (base_url, api_key[:12] if api_key else "")
    if cache_key in _OPENAI_CLIENT_CACHE:
        return _OPENAI_CLIENT_CACHE[cache_key]
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI 兼容模式需要安装 openai 包，请运行：pip install openai"
        ) from e
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    _OPENAI_CLIENT_CACHE[cache_key] = client
    return client


def _is_openai_compat_mode() -> bool:
    """是否启用 OpenAI 兼容模式"""
    return os.getenv("LLM_OPENAI_COMPAT", "").strip() in ("1", "true", "TRUE", "yes", "YES")


def _resolve_api_key() -> str:
    """统一解析 API Key：OpenAI 兼容模式优先读 LLM_API_KEY，否则读 DASHSCOPE_API_KEY"""
    if _is_openai_compat_mode():
        return (
            os.getenv("LLM_API_KEY", "")
            or os.getenv("DASHSCOPE_API_KEY", "")
            or ""
        )
    return os.getenv("DASHSCOPE_API_KEY", "")


def _resolve_base_url() -> str:
    """OpenAI 兼容模式下解析 base_url，默认阿里云 Maas"""
    return os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")


def _resolve_model(default: str = "qwen-plus") -> str:
    """OpenAI 兼容模式下解析模型名"""
    if _is_openai_compat_mode():
        return os.getenv("LLM_MODEL", default)
    return default


# dashscope 仅在非 OpenAI 兼容模式时才需要
if not _is_openai_compat_mode():
    try:
        from dashscope import Generation
    except ImportError:
        Generation = None  # 占位，运行时再校验
else:
    Generation = None


# ============================================================
# 提示词模板
# ============================================================

PROMPT_GENERATE_CANDIDATES = """你是一个专业的领域知识结构化专家。

请从文档集合中建立最多两层的层次化维度 Schema。先归纳 5 至 8 个一级信息轴，
再为每个一级信息轴拆分 2 至 6 个语义独立、可检索的二级叶子维度。
对景区知识库，信息轴通常可以包括“对象与身份、地理位置、历史文化、景观特色、
开放运营、票务规则、交通到达、游客服务”等属性类型；请根据文档选择其中至少 5 个，
不要因为某一个信息轴在示例文本中出现较少就删除它。

严格要求：
1. 一级维度只负责分组，indexable 必须为 false，parent_id 必须为 null。
2. 二级叶子维度必须有明确边界，能够从文本中抽取原子标签，indexable 必须为 true。
3. 一级维度不能混合多个不同类型的信息；每个父维度至少有两个语义独立叶子。
4. 不生成“其他、备注、综合信息”等兜底维度，不把具体实体或标签误当成维度。
   维度必须是“信息的属性类型/字段”，不是文本中出现的具体值；不要把文本中的
   地名、朝代、季节、价格、设施名称直接提升为维度。例如，
   “地理位置”“开放时间”“票务信息”“交通方式”“历史文化”“景观特色”“游客服务”
   可以作为维度；“浙江省”“北京市”“旺季”“淡季”“园林”“古迹”“单程票价”
   只能作为该维度抽取出的标签，绝不能作为维度名称。
5. id 使用稳定的英文或拼音 slug；同一语义的 id 在不同批次中保持稳定。
6. 一级信息轴数量必须为 5 至 8 个；每个一级信息轴至少有两个、最多六个叶子。
7. 只输出合法 JSON，不要 Markdown、解释或思考过程。

请优先输出 6 个一级信息轴，每个一级信息轴输出至少 2 个叶子；叶子名称也必须是
“位置层级、开放时段、票价类型、历史事件、景观组成、交通方式”这类属性类型，
而不是“西湖、旺季、50元、园林”这类具体值。

输出格式必须严格为：
{{"schema_version":"2.0","dimensions":[
  {{"id":"stable_slug","name":"一级名称","parent_id":null,"level":1,"indexable":false,"description":"分组边界"}},
  {{"id":"stable_leaf_slug","name":"叶子名称","parent_id":"stable_slug","level":2,"indexable":true,"description":"可抽取标签的边界"}}
]}}

文档示例（共 {n} 篇，仅供参考）：
---
{docs_snippet}
---
只输出上述 JSON："""

PROMPT_EXTRACT_SINGLE = """你是一个领域知识抽取专家。

给定一个维度的名称，请从以下文档文本中抽取该维度的值。
如果文档中未提及该维度，请返回 null。

维度名称：{dim_name}
文档文本：
---
{text}
---

请直接输出该维度的值（中文），如果未提及则输出 "NULL"：
"""

PROMPT_EXTRACT_VALUES = """你是一个领域知识结构化评估专家。

请判断下面的文本是否包含“{dim_name}”维度，并提取该维度在文本中明确出现的全部值。

严格要求：
1. 只能依据当前文本，不得使用文本外的信息。
2. 一个文本可能有多个值，必须全部列出。
3. 如果没有明确值，返回空数组。
4. 值应尽量使用文本中的原词，不要改写、概括或推断。
5. 每个值必须是互相独立、不可再拆分的原子短语；不得把多个概念拼成一个值。
6. 每个值必须附带支持它的原文 evidence；不允许跨文本推断。
7. 只输出 JSON，不要输出解释或 Markdown。

输出格式：
{{"values": [{{"label":"值1","evidence":"原文证据"}}, {{"label":"值2","evidence":"原文证据"}}]}}

维度名称：{dim_name}
文本：
---
{text}
---
"""

PROMPT_EXTRACT_BATCH = """你是一个领域知识抽取专家。

给定多个二级叶子维度，请从以下文档文本中同时抽取每个维度的全部值。
只输出有明确提及的叶子维度；未提及的维度请忽略。

叶子维度定义（id、名称、边界）：{dims_list}

文档文本：
---
{text}
---

请以 JSON 格式输出，key 必须是叶子维度 id，value 必须是对象数组；每个对象包含 label、evidence、confidence：
{{"leaf_slug_1": [{{"label":"值1","evidence":"原文证据","confidence":0.95}}], "leaf_slug_2": [{{"label":"值2","evidence":"原文证据","confidence":0.90}}]}}
如果某个维度在文档中未提及，请不要包含在输出中。
"""

PROMPT_EXTRACT_MULTI_BATCH = """你是一个领域知识抽取专家。

请分别阅读下面的多条文本，并为每条文本抽取指定维度的值。
只输出文本中有明确依据的维度；没有提及的维度使用空对象。

指定二级叶子维度定义：{dims_list}

输出要求：
1. 只输出 JSON 对象，不要输出解释、Markdown 或思考过程。
2. 顶层 key 必须使用输入记录中的原始 ID。
3. 每个 ID 对应一个对象，对象的 key 必须是指定叶子维度 id，value 必须是对象数组。
4. 每个对象必须有 label、evidence、confidence；label 必须是文本中的原子短语。
5. 一个维度可以返回多个标签，必须全部保留并去重，不得只返回最主要的一个。
6. 不要臆测文本中没有出现的信息，不允许跨 chunk 推断。

输出格式示例：
{{"记录ID-1": {{"place_slug": [{{"label":"衢州","evidence":"原文","confidence":0.95}}]}}, "记录ID-2": {{}}}}

待处理记录：
{records_block}
"""

PROMPT_VALIDATE_SCHEMA_BATCH = """你是一个知识库 schema 验证专家。

当前任务不是给数据库写标签，而是验证“候选维度名称是否适合成为统一的知识库 schema”。
请分别阅读下面的多条 chunk，只判断指定候选维度在每条文本中是否存在明确证据，并列出
文本中原样出现的值，用于统计该维度的覆盖率和区分度。

严格要求：
1. 只能依据当前 chunk 文本，不得跨 chunk 推断或补全。
2. 只允许使用指定候选维度，不要创造新维度。
3. 没有明确值的维度不要输出。
4. 一个 chunk 的同一叶子维度可以有多个值，全部放入对象数组。
5. 每个值附带原文 evidence；不得跨 chunk 推断。
6. 只输出 JSON，不要解释、Markdown 或思考过程。

候选叶子维度定义：{dims_list}

输出格式：
{{"records": [{{"id": "原始ID", "dimensions": {{"leaf_slug": [{{"label":"值1","evidence":"原文","confidence":0.9}}]}}}}]}}

待验证 chunk：
{records_block}
"""

PROMPT_KEYWORDS_FALLBACK = """你是一个关键词提取专家。

请从以下文本中提取 3-5 个最重要的关键词或短语，用于摘要描述。
关键词应反映文本的核心主题和关键信息。

文本：
---
{text}
---

请直接输出关键词列表，用中文逗号分隔：
"""

PROMPT_OPTIMIZE_DIMENSION = """你是一个维度工程专家，负责评估和优化知识维度的质量。

当前需要分析的维度：
- 维度名称：{dim_name}
- 问题类型：{issue_type}
- 诊断数据：{metric_data}
- 抽取样本：{samples}

请根据以上信息做出决策：

问题类型说明：
- "低覆盖率"：该维度只在少数文档中出现
- "低辨识度"：该维度的值太单一，缺乏区分能力
- "语义/数据冗余"：该维度与其他维度高度重叠

决策选项（只能选择其中一个）：
1. KEEP - 保留该维度，即使存在问题但仍有一定价值
2. DELETE - 删除该维度，其信息可被其他维度覆盖
3. RENAME - 重命名该维度，用更准确的概念替代（**不能**改名为"其他"等兜底名称）
4. SPLIT - 拆分为多个更细粒度的维度
5. MERGE - 将该维度与其他维度合并为新的维度（**不能**合并为"其他"等兜底名称）

请以 JSON 格式输出你的决策：
{{"action": "KEEP|DELETE|RENAME|SPLIT|MERGE", "reasoning": "决策理由（50字以内）", "new_nodes": [{{"id":"stable_slug","name":"叶子名称","parent_id":"parent_slug","level":2,"indexable":true,"description":"边界"}}]}}
其中 RENAME 只能返回一个结构化叶子节点；SPLIT 必须返回至少两个结构化叶子节点；MERGE 必须返回合并后的结构化节点。不要返回只有名称的 new_dimensions。
"""

PROMPT_MERGE_WITH_TARGETS = """你是一个维度工程专家，负责评估是否将一个"不达标"的维度与"指定候选目标"中的一个进行合并。

当前需要分析的维度：
- 维度名称：{dim_name}
- 问题类型：{issue_type}
- 诊断数据：{metric_data}
- 抽取样本（前 5 条）：{samples}

【重要】以下是允许的合并目标候选列表（请**只能**从此列表中选择一个作为合并目标，或选择 NOT_MERGE 表示不合并）：
{candidates_block}

任务：
1. 判断"维度 {dim_name}"是否可以合理地合并到候选列表中的某一个维度。
2. 如果可以合并：从候选列表中**精确选择一个目标**（名称必须完全一致），并给出合并后的新维度名称（可以沿用目标名，也可以用更准确的新名称）。
3. 如果都不合适（语义不相关 / 强合会破坏目标维度的纯净度 / 信息确实无价值）：返回 NOT_MERGE。

决策选项（只能选择其中一个）：
1. MERGE - 合并到候选列表中的某个目标维度
2. NOT_MERGE - 不与任何候选合并（保持原样或后续会被删除）

【重要】如果选择 MERGE：
- 合并后的新维度名称**不能**是"其他"、"其他信息"、"备注"、"杂项"等兜底名称
- 新维度必须是有明确语义的具体维度（如"适宜人群"、"疾病类别"等）

请以 JSON 格式输出：
{{"action": "MERGE|NOT_MERGE", "reasoning": "决策理由（50字以内）", "merge_target": "候选列表中的某一个维度名称（仅 MERGE 时填写，必须完全一致）", "new_nodes": [{{"id":"stable_slug","name":"合并后的叶子名称","parent_id":"parent_slug","level":2,"indexable":true,"description":"边界"}}]}}
"""

PROMPT_PARSE_QUERY = """你是一个查询意图理解专家。

给定一个用户查询和可用的层次化维度 Schema，请分析该查询涉及的叶子维度及其对应的全部标签。

Schema（只允许使用这些精确的 dimension id）：
{dims_list}
{enum_info}
{schema_constraint}
用户查询："{query_text}"

请分析：
1. 只选择查询明确表达的维度；优先返回二级叶子，无法细化时才返回一级父维度。
2. 每个约束返回 0 到 N 个互相独立的原子标签。
3. 同一维度多个标签默认 match=ANY。
4. 如果查询表达了某个叶子维度，但没有出现可与候选标签逐字对应的值，保留该约束，
   labels 返回空数组，并用 intent_terms 返回 1 到 8 个原始问题短语，例如“几点开门”、
   “修建了多久”。intent_terms 只表示检索意图，不是数据库标签。
5. 不要把景区名称作为普通维度标签；主体实体由上游单独处理。

请严格输出 JSON：
{{"schema_version":"2.0","constraints":[{{"dimension_id":"leaf_slug","labels":["值1","值2"],"intent_terms":["意图短语"],"match":"ANY"}}]}}
如果查询不涉及维度，constraints 返回空数组。
"""

# v4 保留“只有意图、没有规范标签”的约束。旧版解析器在清洗阶段把
# labels 为空的记录直接丢弃，导致上游只能依赖兜底规则；v4 将 intent_terms
# 作为合法的一等输出，并把它和规范标签明确区分。
PROMPT_PARSE_QUERY_V4 = """你是一个严格的查询结构化解析器。

给定用户查询和可用的二级叶子维度，只能从给定维度中选择查询明确涉及的维度。
不要创建新维度，不要把景区名称当成维度标签。

可用叶子维度（只允许使用其中的 id）：
{dims_list}

用户查询：{query_text}

输出规则：
1. 只输出合法 JSON，不要 Markdown、解释或思考过程。
2. 只输出二级叶子维度（indexable=true）。
3. 一个维度的 labels 必须是问题中明确出现、可与知识库标签对应的原子短语；没有这样的字面值时 labels 必须为空数组。
4. 没有规范标签但有明确查询意图时，仍然必须保留该维度，并把原问题中的意图短语写入 intent_terms，例如“几点开门”“怎么预约”“修建了多久”。
5. 一个维度可以有多个 labels 和多个 intent_terms；必须全部保留、去重，不能只返回第一个，也不能把多个概念拼成一个字符串。
6. 同一维度的多个标签使用 match=ANY；不同维度分别输出。
7. 不跨文档推断，不使用维度清单之外的 id。

输出格式：
{{"schema_version":"4.0","constraints":[{{"dimension_id":"leaf_id","labels":[],"intent_terms":["几点开门"],"match":"ANY"}}]}}

如果没有明确的维度意图，返回 {{"schema_version":"4.0","constraints":[]}}。
只输出 JSON："""


PROMPT_FORMAT_REPAIR = """你是 JSON Schema 修复器。请修复下面的模型输出，使其满足要求：
1. 只输出合法 JSON；2. schema_version 必须为 "2.0"；3. dimensions 是两层树；
4. 一级 indexable=false 且 parent_id=null；5. 二级 indexable=true 且 parent_id 指向一级；
6. 一级节点必须有 5 至 8 个；7. 每个一级至少有两个、最多六个叶子；
8. 维度名称必须是信息属性类型，而不是具体值。比如“地理位置”是维度，
“浙江省”是标签；“开放时间”是维度，“旺季”是标签；“票务信息”是维度，
“单程票价”是标签；9. 不添加“其他、备注、综合信息”等兜底维度。

结构校验错误：{errors}
原始输出：
{raw}
"""


PROMPT_SCHEMA_CONSTRAINT = """
【重要】以下字段名称是 Milvus 数据库中的实际字段名，请务必使用这些精确名称作为 JSON 的 key，切勿自行创造维度名称：
{schema_fields}
"""


def _clip_text(text: str, max_chars: int) -> str:
    """限制提示词长度，同时保留文本首尾，避免只看前缀漏掉后半段维度。"""
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[中间内容省略，仅用于控制长度]...\n{text[-tail:]}"


# ============================================================
# 核心类
# ============================================================

class DimensionMiningWithQwen:
    """
    通义千问驱动的维度挖掘服务。
    封装所有 LLM 调用逻辑，向上游模块提供统一的接口。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            api_key: API Key。若为 None，则从环境变量读取。
                     OpenAI 兼容模式读 LLM_API_KEY，其他模式读 DASHSCOPE_API_KEY。
            model_name: 使用的模型名称。OpenAI 兼容模式默认从 LLM_MODEL 读取，否则默认 qwen-plus。
            base_url: 仅 OpenAI 兼容模式生效。
        """
        # 判断模式
        self.openai_compat = _is_openai_compat_mode()

        if self.openai_compat:
            self.api_key = api_key or _resolve_api_key()
            self.model_name = model_name or _resolve_model(default="qwen-plus")
            self.base_url = (base_url or _resolve_base_url()).rstrip("/")
        else:
            self.api_key = api_key or _resolve_api_key()
            self.model_name = model_name or "qwen-plus"
            self.base_url = None

        if not self.api_key:
            raise ValueError(
                "未找到 API Key。"
                "请设置环境变量 DASHSCOPE_API_KEY 或 LLM_API_KEY，"
                "或在构造函数中传入 api_key。"
            )

        # 通过环境变量控制请求间隔，避免批量维度抽取触发服务限流。
        # 默认 0 保持原有调用速度；服务器实验显式设置为 2 秒。
        try:
            self.api_interval = max(0.0, float(os.getenv("LLM_API_INTERVAL", "0")))
        except ValueError:
            self.api_interval = 0.0
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    def _wait_for_rate_limit(self):
        """在发起下一次 LLM 请求前等待指定间隔。"""
        if self.api_interval <= 0:
            return
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.api_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    # ----------------------------------------------------------
    # 内部调用方法
    # ----------------------------------------------------------

    def _call_llm(self, prompt: str, temperature: float = 0.7, timeout: int = 120) -> str:
        """
        通用的 LLM 调用封装。

        Args:
            prompt: 构造好的提示词。
            temperature: 温度参数，控制随机性。
            timeout: 超时秒数。

        Returns:
            LLM 输出的文本内容。
        """
        self._wait_for_rate_limit()
        if self.openai_compat:
            return self._call_llm_openai(prompt, temperature=temperature, timeout=timeout)

        messages = [{"role": "user", "content": prompt}]
        response = Generation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            result_format="message",
            temperature=temperature,
            top_p=0.9,
            request_timeout=timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM 调用失败 [status={response.status_code}]: "
                f"{response.message}"
            )

        return response.output["choices"][0]["message"]["content"].strip()

    def _call_llm_openai(self, prompt: str, temperature: float = 0.7, timeout: int = 60) -> str:
        """OpenAI 兼容协议调用"""
        client = _get_openai_client(self.base_url, self.api_key)
        try:
            max_tokens = max(64, int(os.getenv("LLM_MAX_TOKENS", "512")))
        except ValueError:
            max_tokens = 512
        request_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 0.9,
            "timeout": timeout,
            "max_tokens": max_tokens,
        }
        # Qwen3 默认可能输出较长的思考过程。维度标签只需要结构化短答案，
        # 关闭思考并设置较小的输出上限可显著降低批量标签生成的延迟。
        if "qwen3" in self.model_name.lower():
            request_kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        resp = client.chat.completions.create(**request_kwargs)
        if not resp.choices:
            raise RuntimeError(f"OpenAI 兼容 LLM 返回空 choices: {resp}")
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _tolerant_json_text(value: str) -> str:
        """Repair common gateway JSON formatting errors conservatively."""

        text = str(value or "").strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            # 如果没有围栏，只保留第一个对象/数组到最后一个闭合符，
            # 去掉模型追加的中文说明。
            starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
            if starts:
                start = min(starts)
                end = max(text.rfind("}"), text.rfind("]"))
                if end > start:
                    text = text[start:end + 1]

        text = re.sub(r"(?m)^\s*//.*$", "", text)
        text = re.sub(r",\s*//.*?(?=\n|$)", ",", text)
        text = re.sub(r",\s*([}\]])", r"\1", text).strip()

        # 修复 evidence 中偶尔出现的未转义双引号和真实换行。判断一个
        # 双引号是否为字符串结束符时，只看后面的 JSON 分隔符，避免把
        # 中文引号或证据文本截断。
        repaired = []
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if escaped:
                repaired.append(char)
                escaped = False
                continue
            if char == "\\":
                repaired.append(char)
                escaped = True
                continue
            if char == '"':
                if not in_string:
                    in_string = True
                    repaired.append(char)
                    continue
                lookahead = text[index + 1:]
                lookahead = re.sub(r"^\s+", "", lookahead)
                if lookahead.startswith((",", ":", "}", "]")):
                    in_string = False
                    repaired.append(char)
                else:
                    repaired.append('\\"')
                continue
            if in_string and char in "\r\n\t":
                repaired.append({"\r": "\\r", "\n": "\\n", "\t": "\\t"}[char])
            else:
                repaired.append(char)
        return "".join(repaired)

    def _call_llm_json(self, prompt: str, temperature: float = 0.3) -> Any:
        """
        调用 LLM 并尝试解析为 JSON 对象。

        Returns:
            解析后的 Python 对象（dict / list）。
        """
        text = self._call_llm(prompt, temperature=temperature)

        json_str = self._tolerant_json_text(text)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 部分网关会把 evidence 中的换行直接写入字符串。JSON
            # 结构仍可明确识别时，允许控制字符存在，避免因此丢弃整批
            # 多标签结果；不会放宽对象字段和维度白名单校验。
            try:
                return json.loads(json_str, strict=False)
            except json.JSONDecodeError:
                pass
            # Qwen 偶尔会连续输出两个 JSON 对象或在 JSON 后附加说明。
            # 用 raw_decode 找到第一个完整对象，避免贪婪正则把两段拼在一起。
            decoder = json.JSONDecoder()
            for match2 in re.finditer(r"[\{\[]", json_str):
                try:
                    value, _ = decoder.raw_decode(json_str[match2.start():])
                    if isinstance(value, (dict, list)):
                        return value
                except json.JSONDecodeError:
                    continue
            raise ValueError(f"LLM 输出无法解析为 JSON:\n{text}")

    def _call_llm_json_with_repair(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        repair_errors: str = "输出不是目标 JSON 结构",
    ) -> Any:
        """Call once, then allow exactly one format-repair retry."""

        try:
            return self._call_llm_json(prompt, temperature=temperature)
        except Exception as first_error:
            raw = str(first_error)
            repair_prompt = PROMPT_FORMAT_REPAIR.format(
                errors=f"{repair_errors}; 首次解析错误: {first_error}",
                raw=raw,
            )
            return self._call_llm_json(repair_prompt, temperature=0.0)

    @staticmethod
    def _dimension_descriptors(dimensions: List[Any]) -> List[Dict[str, Any]]:
        """Normalize names/nodes to compact prompt descriptors."""

        descriptors = []
        for item in dimensions or []:
            if isinstance(item, dict):
                node = dict(item)
                node_id = str(node.get("id") or stable_dimension_id(node.get("name", ""), node.get("parent_id")))
                descriptors.append({
                    "id": node_id,
                    "name": normalize_text(node.get("name", node_id)),
                    "parent_id": normalize_text(node.get("parent_id")) or None,
                    "level": int(node.get("level", 2)),
                    "description": normalize_text(node.get("description", "")),
                    "indexable": bool(node.get("indexable", True)),
                    "aliases": [normalize_text(v) for v in (node.get("aliases", []) or []) if normalize_text(v)],
                    "value_policy": "multi",
                    "max_labels": max(1, int(node.get("max_labels", 5) or 5)),
                })
            else:
                name = normalize_text(item)
                if name:
                    descriptors.append({
                        "id": stable_dimension_id(name),
                        "name": name,
                        "parent_id": None,
                        "level": 2,
                        "description": "兼容旧版维度名称。",
                        "indexable": True,
                        "aliases": [],
                        "value_policy": "multi",
                        "max_labels": 5,
                    })
        return descriptors

    @staticmethod
    def _merge_tag_maps(target: Dict[str, List[Any]], source: Dict[str, Any]) -> None:
        """Merge multi-value responses without dropping later labels."""

        if not isinstance(source, dict):
            return
        for dimension_id, values in source.items():
            if not dimension_id:
                continue
            if isinstance(values, dict):
                values = [values]
            elif isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            bucket = target.setdefault(str(dimension_id).strip(), [])
            seen = {
                normalize_label(v.get("label", "")) if isinstance(v, dict) else normalize_label(v)
                for v in bucket
            }
            for value in values:
                if isinstance(value, dict):
                    label = normalize_text(value.get("label", value.get("value", "")))
                    if not label:
                        continue
                    item = dict(value)
                    item["label"] = label
                    item["raw_label"] = normalize_text(value.get("raw_label", label))
                    item["evidence"] = normalize_text(value.get("evidence", ""))
                else:
                    label = normalize_text(value)
                    if not label:
                        continue
                    item = {
                        "label": label,
                        "raw_label": label,
                        "evidence": "",
                    }
                key = normalize_label(item["label"])
                if key and key not in seen:
                    bucket.append(item)
                    seen.add(key)

    # ============================================================
    # Phase 2: 生成候选维度
    # ============================================================

    @staticmethod
    def _build_candidate_schema(raw: Any) -> Dict[str, Any]:
        """Normalize one candidate response into a strict v2 Schema."""

        if isinstance(raw, list):
            raw = {"schema_version": SCHEMA_VERSION, "dimensions": raw}
        if not isinstance(raw, dict):
            raise SchemaValidationError(["候选维度输出必须是 JSON 对象"])
        raw_nodes = raw.get("dimensions", [])
        if not isinstance(raw_nodes, list):
            raise SchemaValidationError(["dimensions 必须是数组"])

        name_to_id: Dict[str, str] = {}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            name = normalize_text(raw_node.get("name", ""))
            candidate_id = normalize_text(raw_node.get("id", ""))
            parent_raw = normalize_text(raw_node.get("parent_id")) or None
            node_id = candidate_id if is_stable_dimension_id(candidate_id) else stable_dimension_id(name, parent_raw)
            if name:
                name_to_id[name] = node_id

        nodes = []
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            name = normalize_text(raw_node.get("name", ""))
            parent_raw = normalize_text(raw_node.get("parent_id")) or None
            parent_id = name_to_id.get(parent_raw, parent_raw)
            try:
                requested_level = int(raw_node.get("level", 1))
            except (TypeError, ValueError):
                requested_level = 2 if parent_id else 1
            level = 2 if parent_id else requested_level
            candidate_id = normalize_text(raw_node.get("id", ""))
            node_id = candidate_id if is_stable_dimension_id(candidate_id) else stable_dimension_id(name, parent_id)
            nodes.append(make_dimension_node(
                name=name,
                parent_id=parent_id,
                level=level,
                description=raw_node.get("description", ""),
                indexable=(level == 2),
                aliases=raw_node.get("aliases", []),
                max_labels=raw_node.get("max_labels", 5),
                dimension_id=node_id,
            ))
        return {"schema_version": SCHEMA_VERSION, "dimensions": nodes}

    @staticmethod
    def _quality_gate_fallback_schema() -> Dict[str, Any]:
        """Return a safe scenic-information schema when LLM schema repair fails.

        This fallback contains only attribute types, never concrete scenic
        values.  It is deliberately kept at seven parents and two leaves per
        parent so downstream extraction can still use the v2 hierarchy and
        multi-label semantics.  The LLM remains responsible for extracting
        labels from each chunk.
        """

        groups = [
            ("object_identity", "对象与身份", "景区对象、名称和类型属性。", [
                ("entity_name", "对象名称", "文本中明确出现的景区、景点或相关对象名称。"),
                ("scenic_type", "对象类型", "文本中明确出现的景区、景点、遗址或旅游对象类型。"),
            ]),
            ("location", "地理位置", "景区所在区域和空间位置属性。", [
                ("region", "所在地区", "文本中明确出现的省、市、区县或地理区域。"),
                ("location_detail", "位置关系", "文本中明确出现的地址、方位、范围或邻接关系。"),
            ]),
            ("history_culture", "历史文化", "景区历史、文化遗产和相关事件属性。", [
                ("history", "历史沿革", "文本中明确出现的历史时期、沿革、建造或变迁信息。"),
                ("cultural_significance", "文化内涵", "文本中明确出现的文化价值、人物、事件或称号。"),
            ]),
            ("landscape", "景观特色", "景区自然、人文景观和核心游览内容属性。", [
                ("attractions", "核心景点", "文本中明确出现的景点、景观或游览对象名称。"),
                ("feature_composition", "景观组成", "文本中明确出现的建筑、自然地貌、展陈或景观组成。"),
            ]),
            ("operation", "开放运营", "景区开放、营业和季节运营属性。", [
                ("opening_hours", "开放时间", "文本中明确出现的开放、营业或参观时间。"),
                ("seasonal_schedule", "季节安排", "文本中明确出现的旺季、淡季、季节性开放或运营安排。"),
            ]),
            ("ticket_access", "票务与到达", "游客购票、费用和交通到达属性。", [
                ("ticketing", "票务信息", "文本中明确出现的门票、票价、优惠、预约或购票规则。"),
                ("transportation", "交通方式", "文本中明确出现的公交、地铁、自驾、步行或到达路线。"),
            ]),
            ("visitor_service", "游客服务", "游览支持、设施和注意事项属性。", [
                ("facilities", "服务设施", "文本中明确出现的停车、卫生间、餐饮、导览或其他服务设施。"),
                ("visitor_guidance", "游览提示", "文本中明确出现的游览路线、注意事项、限制或服务提示。"),
            ]),
        ]
        nodes = []
        for parent_id, parent_name, parent_description, leaves in groups:
            nodes.append(make_dimension_node(
                name=parent_name,
                parent_id=None,
                level=1,
                description=parent_description,
                indexable=False,
                dimension_id=parent_id,
            ))
            for leaf_id, leaf_name, leaf_description in leaves:
                nodes.append(make_dimension_node(
                    name=leaf_name,
                    parent_id=parent_id,
                    level=2,
                    description=leaf_description,
                    indexable=True,
                    dimension_id=f"{parent_id}.{leaf_id}",
                ))
        return {
            "schema_version": SCHEMA_VERSION,
            "dimensions": nodes,
            "generation_method": "llm_candidate_quality_gate_fallback",
        }

    def generate_candidate_schema(self, docs: List[str]) -> Dict[str, Any]:
        """Generate and validate the v2 two-level candidate schema."""
        if not docs:
            raise ValueError("文档列表为空")

        # 使用分布更均匀的样本，而不是只取列表开头的文档；长度和数量可由
        # 环境变量控制。维度发现阶段若只看 5*300 字，极易漏掉长文本后半段
        # 和少数景区中的有效信息轴。
        try:
            max_docs = max(5, int(os.getenv("DIM_CANDIDATE_DOCS", "20")))
        except ValueError:
            max_docs = 20
        try:
            max_chars = max(300, int(os.getenv("DIM_CANDIDATE_DOC_CHARS", "800")))
        except ValueError:
            max_chars = 800

        sample_count = min(max_docs, len(docs))
        if sample_count == 1:
            sample_indices = [0]
        else:
            sample_indices = [
                round(i * (len(docs) - 1) / (sample_count - 1))
                for i in range(sample_count)
            ]

        snippet_docs = []
        for index in sample_indices:
            doc = str(docs[index] or "")
            snippet_docs.append(_clip_text(doc, max_chars))

        docs_snippet = "\n\n---\n\n".join(snippet_docs)

        prompt = PROMPT_GENERATE_CANDIDATES.format(
            n=len(snippet_docs),
            docs_snippet=docs_snippet
        )

        raw = self._call_llm_json_with_repair(
            prompt,
            temperature=0.7,
            repair_errors="候选维度必须是合法的两层 v2 Schema",
        )
        schema = self._build_candidate_schema(raw)
        errors = validate_schema(schema)
        parents = [node for node in schema.get("dimensions", []) if node.get("level") == 1]
        if not 5 <= len(parents) <= 8:
            errors.append(f"一级信息轴数量必须为 5 至 8 个，当前为 {len(parents)} 个")
        # 这些是本景区语料中常见的标签值，用于拦截模型把值误当成维度。
        # 这不是手工添加维度，只是对 LLM 结构输出做安全校验并触发修复重试。
        forbidden_dimension_values = {
            "浙江省", "北京市", "陕西省", "旺季", "淡季", "园林", "古迹",
            "文化活动", "旅游设施", "单程票价", "押金",
        }
        bad_values = [node.get("name") for node in schema.get("dimensions", [])
                      if node.get("level") == 2 and node.get("name") in forbidden_dimension_values]
        if bad_values:
            errors.append("以下叶子名称看起来是标签值而非维度：" + "、".join(bad_values))
        if errors:
            repair_prompt = PROMPT_FORMAT_REPAIR.format(
                errors="; ".join(errors),
                raw=json.dumps(raw, ensure_ascii=False),
            )
            repaired = self._call_llm_json(repair_prompt, temperature=0.0)
            schema = self._build_candidate_schema(repaired)
            errors = validate_schema(schema)
            parents = [node for node in schema.get("dimensions", []) if node.get("level") == 1]
            if not 5 <= len(parents) <= 8:
                errors.append(f"修复后一级信息轴数量必须为 5 至 8 个，当前为 {len(parents)} 个")
            bad_values = [node.get("name") for node in schema.get("dimensions", [])
                          if node.get("level") == 2 and node.get("name") in forbidden_dimension_values]
            if bad_values:
                errors.append("修复后仍有标签值被误当成维度：" + "、".join(bad_values))
            if errors:
                # The custom gateway may return a semantically plausible but
                # structurally invalid schema even after the one permitted
                # repair retry.  Do not write that schema to an index: switch
                # to the validated attribute-type fallback and keep the LLM
                # multi-label extraction stage active.
                print(
                    "[Warning] LLM v2 Schema 修复仍未通过质量门禁，使用安全属性 Schema："
                    + "; ".join(errors)
                )
                return self._quality_gate_fallback_schema()
        return schema

    def generate_candidate_dimensions(self, docs: List[str]) -> List[str]:
        """Legacy view of v2 generation: return leaf names only."""

        schema = self.generate_candidate_schema(docs)
        return [node["name"] for node in schema["dimensions"] if node.get("level") == 2]

    def extract_dimension_values_with_evidence(
        self,
        text: str,
        dim_name: str,
        max_chars: int = None,
    ) -> List[Dict[str, Any]]:
        """Extract every explicit atomic value and its local evidence."""

        if not text or not dim_name:
            return []
        if max_chars is None:
            try:
                max_chars = max(0, int(os.getenv("DIM_VALIDATION_CHARS", "4000")))
            except ValueError:
                max_chars = 4000
        text = _clip_text(str(text), max_chars)
        prompt = PROMPT_EXTRACT_VALUES.format(dim_name=dim_name, text=text)
        try:
            result = self._call_llm_json(prompt, temperature=0.1)
        except Exception:
            return []

        values = result.get("values", []) if isinstance(result, dict) else result
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        seen = set()
        for raw_value in values:
            if isinstance(raw_value, dict):
                label = normalize_text(raw_value.get("label", raw_value.get("value", "")))
                evidence = normalize_text(raw_value.get("evidence", ""))
                item = dict(raw_value)
            else:
                label = normalize_text(raw_value)
                evidence = ""
                item = {}
            key = normalize_label(label)
            if not key or key.upper() in {"NULL", "NONE", "无", "未提及"} or key in seen:
                continue
            item["label"] = label
            item["raw_label"] = normalize_text(item.get("raw_label", label)) or label
            item["evidence"] = evidence
            if "confidence" in item:
                try:
                    item["confidence"] = float(item["confidence"])
                except (TypeError, ValueError):
                    item.pop("confidence", None)
            cleaned.append(item)
            seen.add(key)
        return cleaned

    def extract_dimension_values(
        self,
        text: str,
        dim_name: str,
        max_chars: int = None,
    ) -> List[str]:
        """Compatibility view returning all labels, never only the first."""

        return [item["label"] for item in self.extract_dimension_values_with_evidence(text, dim_name, max_chars)]

    # ============================================================
    # Phase 3: 单维度单值抽取（迭代验证用）
    # ============================================================

    def extract_dimension_value(self, text: str, dim_name: str) -> Optional[str]:
        """
        从单篇文档中抽取指定维度的单个值。

        Args:
            text: 文档全文。
            dim_name: 维度名称。

        Returns:
            抽取到的值字符串，或 None（表示未命中）。
        """
        values = self.extract_dimension_values(text, dim_name)
        return values[0] if values else None

    # ============================================================
    # Phase 3: 维度优化决策
    # ============================================================

    def optimize_dimension(
        self,
        dim_name: str,
        issue_type: str,
        metric_data: str,
        sample_values: List[str]
    ) -> Dict[str, Any]:
        """
        根据诊断结果，让 LLM 对维度做出优化决策。

        Args:
            dim_name: 维度名称。
            issue_type: 问题类型 ("低覆盖率" | "低辨识度" | "语义/数据冗余")。
            metric_data: 诊断指标数据。
            sample_values: 抽取样本列表。

        Returns:
            {"action": str, "reasoning": str, "new_dimensions": List[str]}
        """
        samples_str = "\n".join(f"  - {v}" for v in sample_values[:10])

        prompt = PROMPT_OPTIMIZE_DIMENSION.format(
            dim_name=dim_name,
            issue_type=issue_type,
            metric_data=metric_data,
            samples=samples_str
        )

        result = self._call_llm_json(prompt, temperature=0.3)
        return result

    def optimize_dimension_v2(
        self,
        node: Dict[str, Any],
        issue_type: str,
        metric_data: str,
        sample_values: List[str],
    ) -> Dict[str, Any]:
        """Return a structured SPLIT/MERGE/RENAME decision for a v2 node."""

        decision = self.optimize_dimension(
            dim_name=str(node.get("name", node.get("id", ""))),
            issue_type=issue_type,
            metric_data=metric_data,
            sample_values=sample_values,
        )
        if not isinstance(decision, dict):
            return {"action": "KEEP", "reasoning": "LLM 返回非对象", "new_nodes": []}
        action = str(decision.get("action", "KEEP")).upper()
        if action not in {"KEEP", "DELETE", "RENAME", "SPLIT", "MERGE"}:
            action = "KEEP"
        raw_nodes = decision.get("new_nodes", [])
        if not isinstance(raw_nodes, list):
            raw_nodes = []
        normalized_nodes = []
        parent_id = normalize_text(node.get("parent_id")) or None
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            name = normalize_text(raw_node.get("name", ""))
            if not name:
                continue
            normalized_nodes.append(make_dimension_node(
                name=name,
                parent_id=normalize_text(raw_node.get("parent_id")) or parent_id,
                level=2,
                description=raw_node.get("description", ""),
                indexable=True,
                aliases=raw_node.get("aliases", []),
                max_labels=raw_node.get("max_labels", node.get("max_labels", 5)),
                dimension_id=normalize_text(raw_node.get("id")) or None,
            ))
        return {
            "action": action,
            "reasoning": normalize_text(decision.get("reasoning", "")),
            "new_nodes": normalized_nodes,
        }

    def merge_with_targets(
        self,
        dim_name: str,
        issue_type: str,
        metric_data: str,
        sample_values: List[str],
        candidate_targets: List[str]
    ) -> Dict[str, Any]:
        """
        让 LLM 在"指定的候选目标列表"中判断能否将 dim_name 与其中一个合并。

        用于"分阶段融合"流程：
          - Stage 1: 传入"合格维度"作为候选
          - Stage 2: 传入"其他不合格维度"作为候选
          - 任何阶段返回 NOT_MERGE 都意味着本阶段融合失败, 可推进到下一阶段

        Args:
            dim_name: 待融合的维度名称。
            issue_type: 问题类型 ("低覆盖率" | "低辨识度")。
            metric_data: 诊断指标数据 (含具体阈值/数值)。
            sample_values: 抽取样本列表。
            candidate_targets: 允许的合并目标候选列表 (必须从中精确选择)。

        Returns:
            {
              "action": "MERGE" | "NOT_MERGE",
              "reasoning": str,
              "merge_target": str,            # 仅 MERGE 时, 必须完全等于 candidate_targets 中某一项
              "new_dimensions": List[str]     # 仅 MERGE 时, 合并后的新维度名称 (1个)
            }
        """
        # 防御: 候选为空直接返回 NOT_MERGE, 不调 LLM
        if not candidate_targets:
            return {
                "action": "NOT_MERGE",
                "reasoning": "候选目标列表为空, 无可融合对象。",
                "merge_target": "",
                "new_dimensions": []
            }

        # 构造候选块 (带编号, 便于 LLM 引用)
        cand_lines = [f"  {i+1}. {name}" for i, name in enumerate(candidate_targets)]
        candidates_block = "\n".join(cand_lines)

        samples_str = "\n".join(f"  - {v}" for v in sample_values[:5])

        prompt = PROMPT_MERGE_WITH_TARGETS.format(
            dim_name=dim_name,
            issue_type=issue_type,
            metric_data=metric_data,
            samples=samples_str,
            candidates_block=candidates_block
        )

        try:
            result = self._call_llm_json(prompt, temperature=0.3)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    [Warning] merge_with_targets JSON 解析失败: {e}")
            return {
                "action": "NOT_MERGE",
                "reasoning": f"LLM 输出解析失败: {e}",
                "merge_target": "",
                "new_dimensions": []
            }

        # 字段规整 + 安全校验
        if not isinstance(result, dict):
            return {
                "action": "NOT_MERGE",
                "reasoning": "LLM 返回非 dict 结构。",
                "merge_target": "",
                "new_dimensions": []
            }

        action = str(result.get("action", "")).upper()
        result["action"] = action

        if action == "MERGE":
            target = str(result.get("merge_target", "")).strip()
            # 安全校验: merge_target 必须在候选列表中, 否则强制 NOT_MERGE
            if target not in candidate_targets:
                print(f"    [Warning] LLM 返回的 merge_target='{target}' 不在候选列表中, 强制 NOT_MERGE")
                result["action"] = "NOT_MERGE"
                result["merge_target"] = ""
                result["new_dimensions"] = []
            else:
                result["merge_target"] = target
                raw_nodes = result.get("new_nodes") or []
                if not isinstance(raw_nodes, list):
                    raw_nodes = []
                if not raw_nodes:
                    # Legacy output is accepted only as an input adapter; the
                    # v2 result always exposes structured nodes as well.
                    old_names = result.get("new_dimensions") or [target]
                    if not isinstance(old_names, list):
                        old_names = [old_names]
                    raw_nodes = [{"name": str(name)} for name in old_names if str(name).strip()]
                result["new_nodes"] = [make_dimension_node(
                    name=node.get("name", ""),
                    parent_id=normalize_text(node.get("parent_id")) or None,
                    level=2,
                    description=node.get("description", ""),
                    indexable=True,
                    aliases=node.get("aliases", []),
                    max_labels=node.get("max_labels", 5),
                    dimension_id=normalize_text(node.get("id")) or None,
                ) for node in raw_nodes if isinstance(node, dict) and normalize_text(node.get("name", ""))]
                result["new_dimensions"] = [node["name"] for node in result["new_nodes"]]
        else:
            # 任何非 MERGE 都归一为 NOT_MERGE
            result["action"] = "NOT_MERGE"
            result["merge_target"] = ""
            result["new_dimensions"] = []

        return result

    # ============================================================
    # TagGenerate: 批量多维度抽取
    # ============================================================

    def _clean_value_map(
        self,
        raw: Any,
        descriptors: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Validate a multi-label response and retain evidence for each label."""

        if not isinstance(raw, dict):
            return {}
        allowed = {}
        max_labels = {}
        for descriptor in descriptors:
            allowed[str(descriptor["id"])] = str(descriptor["id"])
            allowed[str(descriptor["name"])] = str(descriptor["id"])
            max_labels[str(descriptor["id"])] = int(descriptor.get("max_labels", 5) or 5)

        cleaned: Dict[str, List[Dict[str, Any]]] = {}
        for raw_dim, raw_values in raw.items():
            dimension_id = allowed.get(str(raw_dim).strip())
            if not dimension_id:
                continue
            if isinstance(raw_values, dict):
                raw_values = [raw_values]
            elif isinstance(raw_values, str):
                raw_values = [raw_values]
            if not isinstance(raw_values, list):
                continue
            bucket = cleaned.setdefault(dimension_id, [])
            seen = set()
            for raw_value in raw_values:
                if isinstance(raw_value, dict):
                    label = normalize_text(raw_value.get("label", raw_value.get("value", "")))
                    item = dict(raw_value)
                    evidence = normalize_text(raw_value.get("evidence", ""))
                else:
                    label = normalize_text(raw_value)
                    item = {}
                    evidence = ""
                key = normalize_label(label)
                if not key or key in seen:
                    continue
                item["label"] = label
                item["raw_label"] = normalize_text(item.get("raw_label", label)) or label
                item["evidence"] = evidence
                if "confidence" in item:
                    try:
                        item["confidence"] = float(item["confidence"])
                    except (TypeError, ValueError):
                        item.pop("confidence", None)
                bucket.append(item)
                seen.add(key)
                if len(bucket) >= max_labels.get(dimension_id, 5):
                    break
        return {key: values for key, values in cleaned.items() if values}

    def _extract_batch_fallback_v2(
        self,
        text: str,
        descriptors: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback that still calls the plural, evidence-preserving API."""

        results: Dict[str, List[Dict[str, Any]]] = {}
        for descriptor in descriptors:
            values = self.extract_dimension_values_with_evidence(
                text,
                descriptor["name"],
            )
            if values:
                results[str(descriptor["id"])] = values[: int(descriptor.get("max_labels", 5) or 5)]
        return results

    def extract_batch_dimensions_v2(
        self,
        text: str,
        dimensions: List[Any],
        *,
        max_dimensions_per_request: int = 15,
        max_text_chars: int = 1500,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all leaf dimensions in partitions; never silently truncate."""

        if not text or not dimensions:
            return {}
        descriptors = [
            descriptor for descriptor in self._dimension_descriptors(dimensions)
            if descriptor["level"] == 2 and descriptor["indexable"] is True
        ]
        result: Dict[str, List[Dict[str, Any]]] = {}
        limit = max(1, int(max_dimensions_per_request))
        truncated = _clip_text(text, max_text_chars)
        for start in range(0, len(descriptors), limit):
            batch = descriptors[start:start + limit]
            prompt = PROMPT_EXTRACT_BATCH.format(
                dims_list=json.dumps(batch, ensure_ascii=False),
                text=truncated,
            )
            try:
                raw = self._call_llm_json(prompt, temperature=0.1)
                cleaned = self._clean_value_map(raw, batch)
            except Exception as exc:
                print(f"[Warning] v2 批量抽取失败，回退到多值逐维度接口: {exc}")
                cleaned = self._extract_batch_fallback_v2(text, batch)
            self._merge_tag_maps(result, cleaned)
        return result

    def extract_batch_dimensions(
        self,
        text: str,
        dims: List[str]
    ) -> Dict[str, List[str]]:
        """Legacy name-based view backed by the complete v2 multi-value path."""

        descriptors = self._dimension_descriptors(dims)
        extracted = self.extract_batch_dimensions_v2(text, descriptors)
        by_id = {str(node["id"]): str(node["name"]) for node in descriptors}
        return {
            by_id[dimension_id]: [item["label"] for item in values]
            for dimension_id, values in extracted.items()
            if dimension_id in by_id and values
        }

    def extract_multi_chunk_dimensions_v2(
        self,
        records: List[Dict[str, str]],
        dimensions: List[Any],
        max_text_chars: int = 1000,
        *,
        max_dimensions_per_request: int = 15,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Batch chunks by leaf-dimension partitions while retaining all labels."""

        if not records or not dimensions:
            return {}
        descriptors = [
            descriptor for descriptor in self._dimension_descriptors(dimensions)
            if descriptor["level"] == 2 and descriptor["indexable"] is True
        ]
        result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        valid_ids = {str(record.get("doc_id", "")) for record in records}
        record_blocks = []
        for record in records:
            doc_id = str(record.get("doc_id", ""))
            text = _clip_text(str(record.get("doc_text", "")), max_text_chars)
            record_blocks.append(f"ID: {doc_id}\n文本：\n---\n{text}\n---")

        limit = max(1, int(max_dimensions_per_request))
        for start in range(0, len(descriptors), limit):
            batch = descriptors[start:start + limit]
            prompt = PROMPT_EXTRACT_MULTI_BATCH.format(
                dims_list=json.dumps(batch, ensure_ascii=False),
                records_block="\n\n".join(record_blocks),
            )
            try:
                raw_result = self._call_llm_json(prompt, temperature=0.1)
                if not isinstance(raw_result, dict):
                    raise ValueError("多 chunk 返回不是对象")
                for raw_doc_id, raw_tags in raw_result.items():
                    raw_doc_id = str(raw_doc_id)
                    doc_id = raw_doc_id
                    if doc_id not in valid_ids:
                        matches = [candidate for candidate in valid_ids if raw_doc_id.endswith(candidate)]
                        if matches:
                            doc_id = max(matches, key=len)
                    if doc_id not in valid_ids:
                        continue
                    cleaned = self._clean_value_map(raw_tags, batch)
                    if cleaned:
                        self._merge_tag_maps(result.setdefault(doc_id, {}), cleaned)
            except Exception as exc:
                print(f"[Warning] 多 chunk v2 批量抽取失败，回退到逐 chunk 多值接口: {exc}")
                for record in records:
                    doc_id = str(record.get("doc_id", ""))
                    fallback = self.extract_batch_dimensions_v2(
                        str(record.get("doc_text", "")),
                        batch,
                        max_dimensions_per_request=limit,
                        max_text_chars=max_text_chars,
                    )
                    if fallback:
                        self._merge_tag_maps(result.setdefault(doc_id, {}), fallback)
        return result

    def extract_multi_chunk_dimensions(
        self,
        records: List[Dict[str, str]],
        dims: List[str],
        max_text_chars: int = 1000,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Legacy view backed by the complete v2 multi-value path."""

        descriptors = self._dimension_descriptors(dims)
        by_id = {str(node["id"]): str(node["name"]) for node in descriptors}
        raw = self.extract_multi_chunk_dimensions_v2(records, descriptors, max_text_chars)
        return {
            doc_id: {
                by_id[dimension_id]: [item["label"] for item in values]
                for dimension_id, values in tag_map.items()
                if dimension_id in by_id and values
            }
            for doc_id, tag_map in raw.items()
        }

    def validate_dimension_schema_batch_v2(
        self,
        records: List[Dict[str, str]],
        dimensions: List[Any],
        max_text_chars: int = 4000,
        *,
        max_dimensions_per_request: int = 15,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Validate leaf support in batches, retaining all evidence objects."""

        if not records or not dimensions:
            return {}
        descriptors = [
            descriptor for descriptor in self._dimension_descriptors(dimensions)
            if descriptor["level"] == 2 and descriptor["indexable"] is True
        ]
        result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        valid_ids = {str(record.get("doc_id", "")) for record in records}
        records_block = []
        for record in records:
            doc_id = str(record.get("doc_id", ""))
            text = _clip_text(str(record.get("doc_text", "")), max_text_chars)
            records_block.append(f"ID: {doc_id}\n文本：\n---\n{text}\n---")

        limit = max(1, int(max_dimensions_per_request))
        for start in range(0, len(descriptors), limit):
            batch = descriptors[start:start + limit]
            prompt = PROMPT_VALIDATE_SCHEMA_BATCH.format(
                dims_list=json.dumps(batch, ensure_ascii=False),
                records_block="\n\n".join(records_block),
            )
            try:
                raw = self._call_llm_json(prompt, temperature=0.1)
                raw_records = raw.get("records", []) if isinstance(raw, dict) else []
                if not isinstance(raw_records, list):
                    raise ValueError("schema 验证返回 records 不是数组")
                for item in raw_records:
                    if not isinstance(item, dict):
                        continue
                    doc_id = str(item.get("id", item.get("doc_id", ""))).strip()
                    if doc_id not in valid_ids:
                        continue
                    cleaned = self._clean_value_map(item.get("dimensions", {}), batch)
                    if cleaned:
                        self._merge_tag_maps(result.setdefault(doc_id, {}), cleaned)
            except Exception as exc:
                print(f"[Warning] v2 schema 批量验证失败，回退到逐 chunk 多值接口: {exc}")
                for record in records:
                    doc_id = str(record.get("doc_id", ""))
                    fallback = self.extract_batch_dimensions_v2(
                        str(record.get("doc_text", "")),
                        batch,
                        max_dimensions_per_request=limit,
                        max_text_chars=max_text_chars,
                    )
                    if fallback:
                        self._merge_tag_maps(result.setdefault(doc_id, {}), fallback)
        return result

    def validate_dimension_schema_batch(
        self,
        records: List[Dict[str, str]],
        dims: List[str],
        max_text_chars: int = 4000,
    ) -> Dict[str, Dict[str, List[str]]]:
        """Legacy view backed by v2 schema validation."""

        descriptors = self._dimension_descriptors(dims)
        by_id = {str(node["id"]): str(node["name"]) for node in descriptors}
        raw = self.validate_dimension_schema_batch_v2(records, descriptors, max_text_chars)
        return {
            doc_id: {
                by_id[dimension_id]: [item["label"] for item in values]
                for dimension_id, values in tag_map.items()
                if dimension_id in by_id and values
            }
            for doc_id, tag_map in raw.items()
        }

    def _extract_batch_fallback(
        self,
        text: str,
        dims: List[str]
    ) -> Dict[str, List[str]]:
        """
        批量抽取失败时的兜底策略：逐维度调用。
        """
        extracted = self.extract_batch_dimensions_v2(text, dims)
        by_id = {str(node["id"]): str(node["name"]) for node in self._dimension_descriptors(dims)}
        return {
            by_id[dimension_id]: [item["label"] for item in values]
            for dimension_id, values in extracted.items()
            if dimension_id in by_id and values
        }

    # ============================================================
    # TagGenerate: 兜底关键词抽取
    # ============================================================

    def extract_keywords_fallback(self, text: str) -> List[str]:
        """
        当标准维度抽取全部失败时，用关键词兜底。

        Args:
            text: 文档全文。

        Returns:
            关键词列表。
        """
        if not text:
            return []

        truncated = text[:500] if len(text) > 500 else text

        prompt = PROMPT_KEYWORDS_FALLBACK.format(text=truncated)

        try:
            raw = self._call_llm(prompt, temperature=0.5)
            keywords = [k.strip() for k in re.split(r"[，,、]", raw) if k.strip()]
            return keywords[:5]
        except Exception:
            return []

    # ============================================================
    # QueryParser: 查询意图解析
    # ============================================================

    def parse_query_intent_v2(
        self,
        query_text: str,
        dimensions: List[Any],
        enum_values_map: Optional[Dict[str, List[str]]] = None,
        schema_dim_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return versioned constraints with explicit multi-label arrays."""

        if not query_text or not dimensions:
            return {"schema_version": SCHEMA_VERSION, "constraints": []}
        descriptors = self._dimension_descriptors(dimensions)
        allowed = {}
        for node in descriptors:
            allowed[str(node["id"])] = node
            allowed[str(node["name"])] = node
        enum_info = ""
        if enum_values_map:
            enum_info = "候选标签（仅供校验，不可创造）：" + json.dumps(enum_values_map, ensure_ascii=False)
        schema_constraint = ""
        if schema_dim_fields:
            schema_constraint = PROMPT_SCHEMA_CONSTRAINT.format(
                schema_fields="、".join(schema_dim_fields)
            )
        prompt = PROMPT_PARSE_QUERY.format(
            dims_list=json.dumps(descriptors, ensure_ascii=False),
            enum_info=enum_info,
            schema_constraint=schema_constraint,
            query_text=query_text,
        )
        try:
            raw = self._call_llm_json(prompt, temperature=0.3)
        except Exception as exc:
            print(f"[Warning] v2 查询解析失败: {exc}")
            return {"schema_version": SCHEMA_VERSION, "constraints": []}

        raw_constraints = raw.get("constraints", []) if isinstance(raw, dict) else []
        # Accept a legacy mapping only as a compatibility input.
        if isinstance(raw, dict) and not raw_constraints:
            raw_constraints = [
                {"dimension_id": key, "labels": value, "match": "ANY"}
                for key, value in raw.items()
                if key != "schema_version"
            ]
        if not isinstance(raw_constraints, list):
            raw_constraints = []

        constraints = []
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            node = allowed.get(str(item.get("dimension_id", item.get("dimension", ""))).strip())
            if not node:
                continue
            labels = item.get("labels", item.get("values", []))
            if isinstance(labels, str):
                labels = [labels]
            if not isinstance(labels, list):
                continue
            cleaned_labels = []
            seen = set()
            for label in labels:
                label = normalize_text(label)
                key = normalize_label(label)
                if key and key not in seen:
                    cleaned_labels.append(label)
                    seen.add(key)
            if not cleaned_labels:
                continue
            match = str(item.get("match", "ANY")).upper()
            if match not in {"ANY", "ALL"}:
                match = "ANY"
            constraints.append({
                "dimension_id": str(node["id"]),
                "labels": cleaned_labels,
                "match": match,
            })
        return {"schema_version": SCHEMA_VERSION, "constraints": constraints}

    def parse_query_intent_v4(
        self,
        query_text: str,
        dimensions: List[Any],
    ) -> Dict[str, Any]:
        """Parse v4 query constraints without dropping intent-only entries.

        ``labels=[]`` is valid in v4 when ``intent_terms`` contains an explicit
        query cue.  This method is intentionally separate from the v2 method so
        existing experiments keep their exact behaviour and cache namespace.
        """

        if not query_text or not dimensions:
            return {
                "schema_version": "4.0",
                "parser_version": "4.0",
                "constraints": [],
                "diagnostics": {"llm_error": "", "llm_constraint_count": 0},
            }

        descriptors = [
            item for item in self._dimension_descriptors(dimensions)
            if item.get("level") == 2 and item.get("indexable") is True
        ]
        if not descriptors:
            return {
                "schema_version": "4.0",
                "parser_version": "4.0",
                "constraints": [],
                "diagnostics": {"llm_error": "no_indexable_leaf", "llm_constraint_count": 0},
            }

        prompt = PROMPT_PARSE_QUERY_V4.format(
            dims_list=json.dumps(descriptors, ensure_ascii=False),
            query_text=json.dumps(str(query_text), ensure_ascii=False),
        )
        raw: Any = {}
        llm_error = ""
        attempts = 0
        for attempt in range(2):
            attempts += 1
            try:
                raw = self._call_llm_json(prompt, temperature=0.1)
                break
            except Exception as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                # The second attempt only asks for the same compact JSON.  It
                # does not send a broad repair prompt that could invent ids.
                prompt = (
                    PROMPT_PARSE_QUERY_V4.format(
                        dims_list=json.dumps(descriptors, ensure_ascii=False),
                        query_text=json.dumps(str(query_text), ensure_ascii=False),
                    )
                    + "\n再次强调：只输出一个 JSON 对象，不要输出任何其它字符。"
                )

        allowed: Dict[str, Dict[str, Any]] = {}
        for node in descriptors:
            allowed[str(node["id"])] = node
            allowed[normalize_text(node.get("name", ""))] = node
            for alias in node.get("aliases", []) or []:
                alias = normalize_text(alias)
                if alias:
                    allowed[alias] = node

        raw_constraints = raw.get("constraints", []) if isinstance(raw, dict) else []
        if not isinstance(raw_constraints, list):
            raw_constraints = []

        def as_values(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]

        def clean_values(value: Any, limit: int) -> List[str]:
            result: List[str] = []
            seen = set()
            for raw_value in as_values(value):
                value_text = normalize_text(raw_value)
                key = normalize_label(value_text)
                if not key or key in seen or len(key) < 2:
                    continue
                result.append(value_text)
                seen.add(key)
                if len(result) >= limit:
                    break
            return result

        constraints: List[Dict[str, Any]] = []
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            raw_dimension = normalize_text(item.get("dimension_id", item.get("dimension", "")))
            node = allowed.get(raw_dimension)
            if not node:
                continue
            max_labels = max(1, int(node.get("max_labels", 5) or 5))
            labels = clean_values(item.get("labels", item.get("values", [])), max_labels)
            intent_terms = clean_values(item.get("intent_terms", []), 8)
            # This is the key v4 invariant: intent-only constraints survive.
            if not labels and not intent_terms:
                continue
            match = str(item.get("match", "ANY")).upper()
            constraints.append({
                "dimension_id": str(node["id"]),
                "labels": labels,
                "intent_terms": intent_terms,
                "match": match if match in {"ANY", "ALL"} else "ANY",
                "source": "llm_v4",
            })

        return {
            "schema_version": "4.0",
            "parser_version": "4.0",
            "constraints": constraints,
            "diagnostics": {
                "llm_error": llm_error,
                "llm_attempts": attempts,
                "llm_constraint_count": len(constraints),
                "llm_intent_only_count": sum(
                    bool(item.get("intent_terms")) and not item.get("labels")
                    for item in constraints
                ),
            },
        }

    def parse_query_intent(
        self,
        query_text: str,
        dims: List[str],
        enum_values_map: Dict[str, List[str]],
        schema_dim_fields: List[str] = None
    ) -> Dict[str, List[str]]:
        """Legacy name-based view backed by the v2 query parser."""

        descriptors = self._dimension_descriptors(dims)
        by_id = {str(node["id"]): str(node["name"]) for node in descriptors}
        parsed = self.parse_query_intent_v2(
            query_text,
            descriptors,
            enum_values_map=enum_values_map,
            schema_dim_fields=schema_dim_fields,
        )
        return {
            by_id[item["dimension_id"]]: list(item["labels"])
            for item in parsed["constraints"]
            if item["dimension_id"] in by_id
        }


# ============================================================
# 便捷入口（直接 python llm_service.py 可测试连通性）
# ============================================================

if __name__ == "__main__":
    import pprint

    # 从环境变量读取 API Key（也可直接传入）
    miner = DimensionMiningWithQwen(
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        model_name="qwen-plus"
    )

    # 测试 1: 生成候选维度
    print("=== 测试 1: generate_candidate_dimensions ===")
    test_docs = [
        "儿童感冒发热，可使用布洛芬混悬液，每次5ml，每日3次。适用于3-12岁儿童。",
        "老年人腰腿痛，可服用氨基葡萄糖胶囊，每日2次，每次1粒。",
        "孕妇感冒应避免使用布洛芬，建议使用对乙酰氨基酚，并遵医嘱。"
    ]
    dims = miner.generate_candidate_dimensions(test_docs)
    print(f"候选维度: {dims}")

    # 测试 2: 批量抽取
    print("\n=== 测试 2: extract_batch_dimensions ===")
    test_text = "本品适用于3岁以上儿童及成人，用于缓解感冒引起的发热、头痛，鼻塞等症状。儿童用量请遵医嘱。"
    extracted = miner.extract_batch_dimensions(test_text, ["适宜人群", "功效作用", "疾病类别"])
    pprint.pprint(extracted)

    # 测试 3: 查询意图解析
    print("\n=== 测试 3: parse_query_intent ===")
    parsed = miner.parse_query_intent(
        "儿童发烧咳嗽应该怎么办",
        dims=["适宜人群", "疾病类别", "治疗方案", "功效作用"],
        enum_values_map={
            "适宜人群": ["儿童", "成人", "老人", "婴幼儿", "孕妇"],
            "疾病类别": ["呼吸道疾病", "发热相关", "消化系统疾病"]
        }
    )
    pprint.pprint(parsed)
