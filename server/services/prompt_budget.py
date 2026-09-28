"""Priority-aware prompt assembly shared by image and video generation.

Prompt 预算策略（替代历史上的 ``prompt[:6000]`` 末尾硬截断）：

1. 调用方按「风格 → 人物身份 → 场景/构图/首帧 → 动作/情绪/运镜 → 连续性
   规则与低优先级 SOP」从高到低传入具名字段；
2. :func:`assemble_prompt` 只从最低优先级（列表末尾）开始整字段丢弃，
   绝不从字段中间截断，因此身份/动作/情绪/运镜事实保持完整；
3. :func:`ensure_critical_fields` 在极端超预算时兜底：人物身份、动作、
   情绪、运镜等关键字段即使被丢弃也会以精简形式补回。
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def assemble_prompt(
    fields: Iterable[tuple[str, str]],
    *,
    max_chars: int = 6000,
) -> tuple[str, list[str]]:
    """Join prompt fields while dropping low-priority fields first.

    ``fields`` must be ordered from highest to lowest priority. No field is
    cut in the middle, so identity/action/emotion/camera facts remain intact.
    Returns the prompt and names of omitted fields for diagnostics.
    """
    kept: list[str] = []
    dropped: list[str] = []
    used = 0
    values = list(fields)
    for name, value in values:
        text = " ".join(str(value or "").split())
        if not text:
            continue
        addition = text if not kept else ", " + text
        if used + len(addition) <= max_chars:
            kept.append(text)
            used += len(addition)
        else:
            dropped.append(name)
    return ", ".join(kept), dropped


def ensure_critical_fields(
    prompt: str,
    critical: Iterable[tuple[str, str]],
    *,
    max_chars: int = 6000,
) -> tuple[str, list[str]]:
    """Guarantee that critical facts survive budget trimming.

    Any critical field whose value is missing from ``prompt`` is appended in
    compact ``name: value`` form (trimming older low-priority tail content if
    needed). Returns the prompt and the names of re-added fields so callers
    can log them.
    """
    text = str(prompt or "").strip().strip(",")
    if not text:
        text = ""
    readded: list[str] = []
    for name, value in critical:
        fact = " ".join(str(value or "").split())
        if not fact:
            continue
        if _contains_fact(text, fact):
            continue
        readded.append(name)
        addition = f"{name}: {fact}"
        if len(text) + 2 + len(addition) <= max_chars:
            text = f"{text}, {addition}" if text else addition
            continue
        # 预算耗尽：从尾部砍低优先级内容，腾出空间补关键字段。
        while text and len(text) + 2 + len(addition) > max_chars:
            cut = text.rfind(",")
            if cut <= 0:
                text = ""
                break
            text = text[:cut].rstrip().rstrip(",")
        if len(text) + 2 + len(addition) <= max_chars:
            text = f"{text}, {addition}" if text else addition
    return text, readded


def _contains_fact(prompt: str, fact: str) -> bool:
    """Case-insensitive containment check tolerant of comma spacing."""

    def normalize(value: str) -> str:
        return re.sub(r"[\s,]+", " ", value).strip().casefold()

    return normalize(fact) in normalize(prompt)


def dedupe_terms(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        # Negative prompts are commonly supplied as comma/semicolon-delimited
        # phrases. Normalize at term granularity so conflict removal can catch
        # an accidental positive "anime" even when it came from a longer list.
        for raw_term in re.split(r"[,;，；]", str(value or "")):
            term = " ".join(raw_term.split()).strip(" ,")
            key = term.casefold()
            if term and key not in seen:
                result.append(term)
                seen.add(key)
    return result


def remove_conflicting_terms(positive: str, negative: Iterable[str]) -> tuple[str, list[str]]:
    """Remove exact negative terms accidentally repeated in positive text."""
    removed: list[str] = []
    result = positive
    for term in dedupe_terms(negative):
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", result, flags=re.IGNORECASE):
            result = re.sub(rf"(?<!\w){re.escape(term)}(?!\w),?\s*", "", result, flags=re.IGNORECASE)
            removed.append(term)
    return result.strip(" ,"), removed
