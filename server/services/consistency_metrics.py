"""一致性结果的可测量指标与 VLM 验证。

请求载荷指标（参考覆盖率、角色/场景/连续性角色覆盖、实际控制类型）是确定性
指标；视觉结果再用已配置的 VLM 做一次可复核的评分。VLM 不可用时结果明确标为
``unavailable``，绝不把提示词或结构检查冒充成视觉一致性通过。
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from config import settings

DEFAULT_THRESHOLDS = {
    "character_identity": 0.78,
    "scene_group": 0.78,
    "shot_continuity": 0.72,
}


def payload_metrics(
    *,
    references_validated: int,
    references_sent: list[dict[str, Any]] | int,
    required_roles: list[str] | None = None,
    control_types_sent: list[str] | None = None,
    provider_capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """由真实请求载荷派生确定性指标。"""

    sent_items = references_sent if isinstance(references_sent, list) else [
        {"type": "reference_image", "index": index} for index in range(int(references_sent or 0))
    ]
    sent_roles = {str(item.get("role") or item.get("type") or "") for item in sent_items if isinstance(item, dict)}
    required = [str(item) for item in (required_roles or []) if item]
    covered = [role for role in required if role in sent_roles]
    validated = max(0, int(references_validated or 0))
    sent_count = len(sent_items)
    return {
        "method": "request_payload",
        "reference_coverage": {
            "validated": validated,
            "sent": sent_count,
            "ratio": round(sent_count / validated, 4) if validated else 0.0,
        },
        "role_coverage": {
            "required": required,
            "sent": sorted(sent_roles),
            "covered": covered,
            "ratio": round(len(covered) / len(required), 4) if required else 1.0,
        },
        "control_types_sent": list(dict.fromkeys(control_types_sent or [])),
        "provider_capabilities": provider_capabilities or {},
    }


def _contact_sheet(paths: list[str], destination: Path) -> bool:
    images: list[Image.Image] = []
    try:
        for raw in paths:
            path = Path(raw)
            if not path.exists() or not path.is_file():
                continue
            with Image.open(path) as image:
                image.load()
                images.append(image.convert("RGB"))
        if not images:
            return False
        thumb_width = 512
        thumbs = []
        for image in images:
            ratio = thumb_width / max(1, image.width)
            thumbs.append(image.resize((thumb_width, max(1, int(image.height * ratio)))))
        height = max(image.height for image in thumbs)
        sheet = Image.new("RGB", (thumb_width * len(thumbs), height), "white")
        for index, image in enumerate(thumbs):
            sheet.paste(image, (index * thumb_width, 0))
        sheet.save(destination, format="PNG")
        return True
    except Exception:
        return False
    finally:
        for image in images:
            image.close()


def _parse_vlm_json(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    match = re.search(r"\{.*\}", value, flags=re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


async def validate_visual_consistency(
    *,
    generated_path: str,
    reference_paths: list[str],
    metric_keys: tuple[str, ...] = ("character_identity", "scene_group", "shot_continuity"),
) -> dict[str, Any]:
    """用 VLM 评估生成图与参考素材；不可用时返回诚实的 unavailable 报告。"""

    mode = str(getattr(settings, "CONSISTENCY_VALIDATION_MODE", "vlm") or "vlm").strip().lower()
    if mode in {"off", "disabled", "none"}:
        return {"method": "vlm", "status": "disabled", "scores": {}, "thresholds": DEFAULT_THRESHOLDS}

    try:
        from services.llm_service import LLMService

        validator = LLMService()
        if not validator.available:
            return {"method": "vlm", "status": "unavailable", "reason": "未配置可验证视觉一致性的 VLM", "scores": {}}
    except Exception as exc:  # noqa: BLE001
        return {"method": "vlm", "status": "unavailable", "reason": str(exc)[:200], "scores": {}}

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        contact_path = Path(handle.name)
    try:
        if not _contact_sheet([generated_path, *reference_paths], contact_path):
            return {"method": "vlm", "status": "unavailable", "reason": "没有可读的生成图或参考图", "scores": {}}
        prompt = (
            "Compare the generated image with the reference sheet(s) for a comic/video production pipeline. "
            "Return strict JSON only with numeric scores from 0 to 1 for "
            + ", ".join(metric_keys)
            + ", plus a short evidence field. These scores are visual verification results, not prompt compliance claims."
        )
        raw = await validator.call_with_image(prompt, str(contact_path))
        parsed = _parse_vlm_json(raw)
        scores = {}
        for key in metric_keys:
            try:
                scores[key] = max(0.0, min(1.0, float(parsed.get(key))))
            except (TypeError, ValueError):
                scores[key] = None
        passing = all(value is not None and value >= DEFAULT_THRESHOLDS.get(key, 0.75) for key, value in scores.items())
        return {
            "method": "vlm",
            "status": "passed" if passing else "needs_review",
            "scores": scores,
            "thresholds": {key: DEFAULT_THRESHOLDS.get(key, 0.75) for key in metric_keys},
            "evidence": str(parsed.get("evidence") or "")[:500],
        }
    except Exception as exc:  # noqa: BLE001
        return {"method": "vlm", "status": "unavailable", "reason": str(exc)[:200], "scores": {}}
    finally:
        contact_path.unlink(missing_ok=True)


def combine_report(payload: dict[str, Any], visual: dict[str, Any]) -> dict[str, Any]:
    """把确定性载荷指标与 VLM 结果合成任务报告。"""

    return {
        **dict(payload or {}),
        "visual_validation": dict(visual or {}),
        "measurable": True,
        "claim_scope": "request_payload_and_vlm" if (visual or {}).get("status") == "passed" else "request_payload_only",
    }
