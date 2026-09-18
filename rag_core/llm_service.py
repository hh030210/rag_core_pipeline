"""原始 code_jyx 维度 LLM 服务的兼容导出。

生产实现位于 ``code_jyx.llm_service``；新项目只保留这个导入兼容层。
"""

from code_jyx.llm_service import *  # noqa: F401,F403
