#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于规则的机械去噪器。

这个模块是现有 PPL 去噪的独立替代方案：只使用字符串、正则表达式和重复
统计，不调用 LLM、Embedding 或语言模型。规则无法确定时默认保留内容。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List


class MechanicalDenoiser:
    """对句子列表做保守的、可审计的规则去噪。"""

    _NAV_RE = re.compile(
        r"^(?:首页|上一页|下一页|返回顶部|返回首页|返回上一页|打印|分享|收藏|登录|注册|"
        r"阅读全文|阅读原文|点击查看更多|点击下载|网站导航|目录|免责声明|版权声明)$"
    )
    _PAGE_RE = re.compile(
        r"^(?:第\s*[0-9一二三四五六七八九十百]+\s*页|"
        r"[\-_—–·\s]*[0-9]+[\-_—–·\s]*页?|"
        r"页码\s*[:：]?\s*[0-9]+)$",
        re.IGNORECASE,
    )
    _SEPARATOR_RE = re.compile(r"^[\s\-_=—–·•*~.。！!#]{3,}$")
    _URL_ONLY_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
    _HTML_ONLY_RE = re.compile(r"^</?[A-Za-z][^>]*>$")
    _HEADING_RE = re.compile(
        r"^(?:#{1,6}\s+|第\s*[0-9一二三四五六七八九十百]+\s*[章节篇部分]|"
        r"[0-9一二三四五六七八九十百]+[、.．])"
    )
    _FACT_RE = re.compile(
        r"(?:景区|景点|门票|票价|价格|费用|免费|收费|开放|营业|售票|预约|"
        r"地址|电话|联系方式|交通|公交|地铁|路线|入口|出口|站|码头|"
        r"建议游玩|游玩时间|最佳时间|最佳季节|面积|海拔|高度|距离|"
        r"公里|千米|公顷|米|元|人次|小时|分钟|年|月|日|时|分)",
        re.IGNORECASE,
    )
    _NUMBER_UNIT_RE = re.compile(
        r"(?:\d+(?:\.\d+)?\s*(?:元|公里|千米|米|公顷|小时|分钟|人|人次|路|号|岁)|"
        r"\d{2,4}\s*[年\-/]\s*\d{1,2}(?:\s*[月\-/]\s*\d{1,2})?|"
        r"\d{1,2}\s*[:：]\s*\d{1,2})"
    )
    _CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]")
    _REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")
    _REPEATED_BLOCK_RE = re.compile(r"(.{1,8})\1{2,}")
    _WEB_NOISE_RE = re.compile(
        r"(?:扫码关注|关注公众号|点击关注|下载客户端|下载APP|客户端打开|"
        r"广告|推广链接|隐私政策|用户协议|意见反馈)",
        re.IGNORECASE,
    )

    def __init__(self, duplicate_min_chars: int = 6):
        self.duplicate_min_chars = duplicate_min_chars

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", text or "")
        text = text.replace("\u200b", "").replace("\ufeff", "")
        return re.sub(r"\s+", "", text).strip()

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = MechanicalDenoiser._normalize(text).lower()
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", normalized)

    @staticmethod
    def _meaningful_chars(text: str) -> List[str]:
        return re.findall(r"[0-9A-Za-z\u4e00-\u9fff]", text or "")

    @staticmethod
    def _is_whole_repeated_block(text: str) -> bool:
        """只识别整段由同一字符/短片段重复构成的内容。

        不能用 ``search`` 判断任意子串重复，否则正常历史文本中的引号、
        省略号或固定词组也可能被误删。
        """
        if not text or len(text) < 3:
            return False
        for block_len in range(1, min(8, len(text) // 3) + 1):
            if len(text) % block_len:
                continue
            repeat_count = len(text) // block_len
            if repeat_count >= 3 and text == text[:block_len] * repeat_count:
                return True
        return False

    def _is_protected(self, text: str) -> bool:
        """保护标题、实体和运营事实，避免短句规则误删有效信息。"""
        normalized = self._normalize(text)
        if not normalized:
            return False
        if self._HEADING_RE.search(normalized):
            return True
        if self._FACT_RE.search(normalized) or self._NUMBER_UNIT_RE.search(normalized):
            return True
        # 三到六字的中文专名/建筑名不按长度删除；这类内容常见于景区标题。
        if (
            3 <= len(self._meaningful_chars(normalized)) <= 12
            and re.fullmatch(r"[\u4e00-\u9fff·（）()]+", normalized)
            and not self._REPEATED_CHAR_RE.search(normalized)
            and not self._is_whole_repeated_block(normalized)
        ):
            return True
        return False

    def _hard_noise_reason(self, text: str, protected: bool) -> str:
        normalized = self._normalize(text)
        if not normalized:
            return "空白内容"
        if self._CONTROL_RE.search(text):
            bad_count = len(self._CONTROL_RE.findall(text))
            if bad_count / max(len(text), 1) >= 0.3 or len(self._meaningful_chars(text)) <= 2:
                return "控制字符或乱码"
        if self._NAV_RE.fullmatch(normalized):
            return "网页导航或操作文本"
        if self._PAGE_RE.fullmatch(normalized):
            return "页码文本"
        if self._SEPARATOR_RE.fullmatch(normalized):
            return "分隔线或装饰符号"
        if self._HTML_ONLY_RE.fullmatch(normalized):
            return "孤立 HTML 标签"
        if self._URL_ONLY_RE.fullmatch(normalized) and not protected:
            return "孤立网页链接"
        if self._WEB_NOISE_RE.search(normalized) and not protected:
            return "网页推广或模板文本"

        meaningful = self._meaningful_chars(normalized)
        if not meaningful:
            return "无有效字符"

        if self._is_whole_repeated_block(normalized) and not protected:
            return "连续重复字符"

        punctuation_count = sum(
            1 for ch in normalized if not re.match(r"[0-9A-Za-z\u4e00-\u9fff\s]", ch)
        )
        punctuation_ratio = punctuation_count / max(len(normalized), 1)
        if punctuation_ratio >= 0.85 and len(meaningful) <= 2 and not protected:
            return "标点或符号比例过高"

        # 不用长度单独删除：仅当极短内容同时缺少实体、事实和有效字符时才删除。
        if len(meaningful) <= 1 and not protected:
            return "无独立信息的残片"
        return ""

    def denoise(self, sentences: List[str]) -> Dict:
        """返回保留句子、删除明细和汇总统计。"""
        kept: List[str] = []
        removed: List[Dict] = []
        review: List[Dict] = []
        seen: Dict[str, int] = {}

        for index, raw in enumerate(sentences):
            text = (raw or "").strip()
            protected = self._is_protected(text)
            reason = self._hard_noise_reason(text, protected)
            if reason:
                removed.append({"index": index, "text": text, "reason": reason})
                continue

            fingerprint = self._fingerprint(text)
            if fingerprint and len(fingerprint) >= self.duplicate_min_chars:
                if fingerprint in seen:
                    removed.append({
                        "index": index,
                        "text": text,
                        "reason": "同一子文件内的重复句",
                        "first_index": seen[fingerprint],
                    })
                    continue
                seen[fingerprint] = index

            # 短句不自动删除，只进入审计列表，最终仍然保留。
            meaningful_len = len(self._meaningful_chars(text))
            if meaningful_len <= 5 and not protected:
                review.append({
                    "index": index,
                    "text": text,
                    "reason": "短文本，规则无法确认是否为标题或实体",
                })
            kept.append(text)

        # 极端情况下避免规则把整个子文件清空，完整回退原文。
        if sentences and not kept:
            kept = [s.strip() for s in sentences if (s or "").strip()]
            removed = []
            review.append({
                "index": -1,
                "text": "",
                "reason": "全部句子被判为噪声，已回退保留原文",
            })

        input_chars = sum(len(s or "") for s in sentences)
        kept_chars = sum(len(s) for s in kept)
        removed_chars = sum(len(item["text"]) for item in removed)
        return {
            "kept_sentences": kept,
            "removed": removed,
            "review": review,
            "summary": {
                "input_sentence_count": len(sentences),
                "kept_sentence_count": len(kept),
                "removed_sentence_count": len(removed),
                "review_sentence_count": len(review),
                "input_chars": input_chars,
                "kept_chars": kept_chars,
                "removed_chars": removed_chars,
                "removed_char_ratio": round(removed_chars / max(input_chars, 1), 6),
            },
        }
