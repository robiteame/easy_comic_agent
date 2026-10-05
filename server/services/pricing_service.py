"""模型价格配置与整数金额计算。

记账口径（全项目统一，禁止浮点金额）：

- 金额一律是「货币最小单位的整数倍」：本项目的记账精度为 10^-6 个货币单位
  （micro），1 CNY = 1_000_000 micro。任何一步都用整数运算（``ceil_div``），
  绝不出现 float 金额；
- 计价数量也是整数：llm=输入 token，image=张，video=秒，tts=字符，ffmpeg=秒，
  再按能力对应的「一个计价单位包含多少基础数量」（unit_scale）换算；
- 舍入规则统一为「向上取整到 1 micro」，保证小额调用不会被舍成 0 而丢失成本；
- 没有可用价目时不计算金额：cost_known=False 且 cost_micro=None（前端显示
  「成本未知」）。绝不按 0 元或猜测价格入账。

价格匹配优先级（从具体到宽泛）：model 精确 > provider 通用价（model 为空）>
能力兜底价（provider 与 model 均为空）。只有 configured=True 且单价非 NULL 的
行才参与计费；其余行只是前端展示用的占位模板。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from models import PricingConfig
from models.pricing import DEFAULT_CURRENCY, MICRO_PER_UNIT
from services.providers.usage import (
    CAPABILITIES,
    CAPABILITY_BASE_UNITS,
    CAPABILITY_FFMPEG,
    CAPABILITY_IMAGE,
    CAPABILITY_LABELS,
    CAPABILITY_LLM,
    CAPABILITY_PRICING_UNITS,
    CAPABILITY_SECONDARY_UNITS,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO,
)

# 成本来源，与 UsageRecord.cost_source 取值保持一致。
COST_SOURCE_PRICING = "pricing"
COST_SOURCE_LOCAL = "local"
COST_SOURCE_UNKNOWN = "unknown"

# 分辨率倍率的默认值（1_000_000 = 1.0）。
UNIT_MULTIPLIER_MICRO = 1_000_000


def ceil_div(numerator: int, denominator: int) -> int:
    """整数向上取整除法；除数为 0 或负数时返回 0（调用方应先校验）。"""

    divisor = int(denominator)
    if divisor <= 0:
        return 0
    value = int(numerator)
    if value <= 0:
        return 0
    return -(-value // divisor)


def apply_multiplier(price_micro: int, multiplier_micro: int) -> int:
    """按整数倍率换算单价（向上取整到 1 micro）。"""

    return ceil_div(int(price_micro) * max(0, int(multiplier_micro)), UNIT_MULTIPLIER_MICRO)


def compute_cost_micro(
    quantity: int,
    secondary_quantity: int,
    *,
    unit_price_micro: int | None,
    unit_price_secondary_micro: int | None = None,
    unit_scale: int = 1,
    multiplier_micro: int = UNIT_MULTIPLIER_MICRO,
) -> int | None:
    """按整数算术算出金额（micro）；单价缺失时返回 None（成本未知）。"""

    if unit_price_micro is None:
        return None
    scale = max(1, int(unit_scale))
    total = 0
    if quantity:
        total += ceil_div(
            int(quantity) * apply_multiplier(int(unit_price_micro), multiplier_micro),
            scale,
        )
    if secondary_quantity and unit_price_secondary_micro is not None:
        total += ceil_div(
            int(secondary_quantity) * apply_multiplier(int(unit_price_secondary_micro), multiplier_micro),
            scale,
        )
    return total


def format_amount_micro(amount_micro: int | None, currency: str = DEFAULT_CURRENCY) -> str:
    """把整数 micro 金额格式化成人类可读字符串（仅用于日志/文档，前端自行格式化）。"""

    if amount_micro is None:
        return "成本未知"
    sign = "-" if int(amount_micro) < 0 else ""
    value = abs(int(amount_micro))
    whole, fraction = divmod(value, MICRO_PER_UNIT)
    return f"{sign}{currency} {whole}.{fraction:06d}"


def parse_multipliers(raw: Any) -> dict[str, int]:
    """解析分辨率倍率；只接受「字符串 -> 整数 micro」的映射，拒绝浮点。"""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return {}
    if not isinstance(raw, dict):
        return {}
    multipliers: dict[str, int] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name or isinstance(value, bool):
            continue
        if isinstance(value, float):
            raise ValueError(f"倍率必须是整数 micro（1_000_000 = 1.0）: {name}")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"倍率必须是整数 micro（1_000_000 = 1.0）: {name}") from exc
        if number < 0:
            raise ValueError(f"倍率不能为负: {name}")
        multipliers[name] = number
    return multipliers


def multiplier_for(multipliers: dict[str, int], resolution: str) -> int:
    """按分辨率取倍率；未命中时用 default，再退回 1.0。"""

    key = str(resolution or "").strip()
    if key and key in multipliers:
        return multipliers[key]
    lowered = key.lower()
    for name, value in multipliers.items():
        if name.lower() == lowered:
            return value
    if "default" in multipliers:
        return multipliers["default"]
    return UNIT_MULTIPLIER_MICRO


@dataclass(frozen=True)
class PriceResolution:
    """一次用量匹配到的价目结果。"""

    capability: str
    provider: str = ""
    model: str = ""
    currency: str = DEFAULT_CURRENCY
    unit_scale: int = 1
    unit_price_micro: int | None = None
    unit_price_secondary_micro: int | None = None
    multiplier_micro: int = UNIT_MULTIPLIER_MICRO
    configured: bool = False
    matched_pricing_id: str = ""

    @property
    def priced(self) -> bool:
        return self.unit_price_micro is not None

    def cost_micro(self, quantity: int, secondary_quantity: int = 0) -> int | None:
        if not self.priced:
            return None
        return compute_cost_micro(
            quantity,
            secondary_quantity,
            unit_price_micro=self.unit_price_micro,
            unit_price_secondary_micro=self.unit_price_secondary_micro,
            unit_scale=self.unit_scale,
            multiplier_micro=self.multiplier_micro,
        )

    def snapshot(self) -> dict[str, Any]:
        """写入 UsageRecord 的计费快照（不含任何密钥）。"""

        return {
            "capability": self.capability,
            "provider": self.provider,
            "model": self.model,
            "currency": self.currency,
            "unit_scale": self.unit_scale,
            "unit_price_micro": self.unit_price_micro,
            "unit_price_secondary_micro": self.unit_price_secondary_micro,
            "multiplier_micro": self.multiplier_micro,
            "pricing_id": self.matched_pricing_id,
        }


def _rows_for(db: Session, capability: str) -> list[PricingConfig]:
    return db.query(PricingConfig).filter(PricingConfig.capability == capability).all()


def _rank(row: PricingConfig, provider: str, model: str) -> int:
    """匹配打分：越大越具体。-1 表示不匹配。"""

    row_provider = str(row.provider or "")
    row_model = str(row.model or "")
    if row_model and row_model != model:
        return -1
    if row_provider and row_provider != provider:
        return -1
    if row_model and row_provider:
        return 3
    if row_provider:
        return 2
    if row_model:
        return 1
    return 0


def resolve_price(
    db: Session,
    capability: str,
    provider: str = "",
    model: str = "",
    *,
    resolution: str = "",
) -> PriceResolution:
    """解析某能力 / provider / 模型的生效价目。"""

    capability = str(capability or "").strip().lower()
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    unit_scale = CAPABILITY_PRICING_UNITS.get(capability, ("", 1))[1]
    resolution_holder = PriceResolution(capability=capability, provider=provider, model=model, unit_scale=unit_scale)
    if capability not in CAPABILITIES:
        return resolution_holder

    best: PricingConfig | None = None
    best_rank = -1
    for row in _rows_for(db, capability):
        rank = _rank(row, provider, model)
        if rank < 0:
            continue
        if best is None or rank > best_rank:
            best, best_rank = row, rank
    if best is None:
        return resolution_holder

    multipliers = parse_multipliers(best.resolution_multipliers)
    price = best.unit_price_micro if best.configured else None
    secondary = best.unit_price_secondary_micro if best.configured else None
    return PriceResolution(
        capability=capability,
        provider=provider or str(best.provider or ""),
        model=model or str(best.model or ""),
        currency=str(best.currency or DEFAULT_CURRENCY),
        unit_scale=unit_scale,
        unit_price_micro=int(price) if price is not None else None,
        unit_price_secondary_micro=int(secondary) if secondary is not None else None,
        multiplier_micro=multiplier_for(multipliers, resolution),
        configured=bool(best.configured),
        matched_pricing_id=str(best.id),
    )


# ---------------------------------------------------------------------------
# 价目表读写（系统设置 → 模型价格配置）
# ---------------------------------------------------------------------------

# 出厂模板：只声明「可以配哪些价」，全部 configured=False，价格为空。
# 这样默认安装不会凭空产生金额，界面会明确显示「未配置价格 → 成本未知」。
DEFAULT_PRICING_PROVIDERS: dict[str, tuple[str, ...]] = {
    CAPABILITY_LLM: ("openai-chat",),
    CAPABILITY_IMAGE: ("ark-seedream", "qwen-image", "stability", "placeholder"),
    CAPABILITY_VIDEO: ("ark-seedance", "native-audio", "dashscope-wanx"),
    CAPABILITY_TTS: ("mimo-tts", "tencent-tts", "dashscope-tts"),
    CAPABILITY_FFMPEG: ("local",),
}

# 本地零成本能力的出厂设置：数量照记，但不产生外部费用。
LOCAL_ZERO_COST_PROVIDERS: tuple[tuple[str, str], ...] = (
    (CAPABILITY_IMAGE, "placeholder"),
    (CAPABILITY_FFMPEG, "local"),
)


def _row_dto(row: PricingConfig) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "capability": str(row.capability),
        "provider": str(row.provider or ""),
        "model": str(row.model or ""),
        "currency": str(row.currency or DEFAULT_CURRENCY),
        "unit_price_micro": int(row.unit_price_micro) if row.unit_price_micro is not None else None,
        "unit_price_secondary_micro": (
            int(row.unit_price_secondary_micro) if row.unit_price_secondary_micro is not None else None
        ),
        "resolution_multipliers": parse_multipliers(row.resolution_multipliers),
        "configured": bool(row.configured),
        "note": str(row.note or ""),
        "updated_at": row.updated_at.isoformat() if isinstance(row.updated_at, datetime) else None,
    }


def seed_pricing_defaults(db: Session) -> int:
    """补齐出厂模板行（幂等）；返回新增的行数。"""

    created = 0
    for capability, providers in DEFAULT_PRICING_PROVIDERS.items():
        for provider in providers:
            exists = (
                db.query(PricingConfig)
                .filter(
                    PricingConfig.capability == capability,
                    PricingConfig.provider == provider,
                    PricingConfig.model == "",
                )
                .first()
            )
            if exists is not None:
                continue
            free = (capability, provider) in LOCAL_ZERO_COST_PROVIDERS
            db.add(
                PricingConfig(
                    id=uuid.uuid4().hex,
                    capability=capability,
                    provider=provider,
                    model="",
                    currency=DEFAULT_CURRENCY,
                    unit_price_micro=0 if free else None,
                    unit_price_secondary_micro=None,
                    resolution_multipliers="{}",
                    configured=bool(free),
                    note="本地能力，不产生外部费用" if free else "",
                )
            )
            created += 1
    if created:
        db.commit()
    return created


def list_pricing(db: Session) -> dict[str, Any]:
    """价目表（按能力分组），供系统设置回填。"""

    rows = db.query(PricingConfig).all()
    grouped: dict[str, list[dict[str, Any]]] = {capability: [] for capability in CAPABILITIES}
    for row in rows:
        grouped.setdefault(str(row.capability), []).append(_row_dto(row))
    for _capability, items in grouped.items():
        items.sort(key=lambda item: (item["provider"], item["model"]))
    capabilities = []
    for capability in CAPABILITIES:
        label, scale = CAPABILITY_PRICING_UNITS.get(capability, (capability, 1))
        capabilities.append(
            {
                "capability": capability,
                "label": CAPABILITY_LABELS.get(capability, capability),
                "base_unit": CAPABILITY_BASE_UNITS.get(capability, ""),
                "pricing_unit": label,
                "unit_scale": scale,
                "secondary_unit": CAPABILITY_SECONDARY_UNITS.get(capability),
                "items": grouped.get(capability, []),
            }
        )
    return {
        "currency": _dominant_currency(rows),
        "capabilities": capabilities,
        "micro_per_unit": MICRO_PER_UNIT,
        "rounding": "按整数 micro 向上取整，绝不使用浮点金额",
    }


def _dominant_currency(rows: list[PricingConfig]) -> str:
    for row in rows:
        if row.currency:
            return str(row.currency)
    return DEFAULT_CURRENCY


def _require_int(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} 必须是整数金额（最小货币单位）")
    if isinstance(value, float):
        raise ValueError(f"{field} 必须是整数金额（最小货币单位），不接受小数：{value}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是整数金额（最小货币单位）") from exc
    if str(value).strip() not in {str(number), f"+{number}"} and not str(value).strip().lstrip("-").isdigit():
        raise ValueError(f"{field} 必须是整数金额（最小货币单位）")
    if number < 0:
        raise ValueError(f"{field} 不能为负数")
    return number


def save_pricing(db: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """保存价目表（按 capability/provider/model 合并写入）。"""

    items = payload.get("items") if isinstance(payload, dict) else None
    if items is None:
        items = payload
    if not isinstance(items, list):
        raise ValueError("items 必须是数组")
    currency = str((payload or {}).get("currency") or DEFAULT_CURRENCY).strip() or DEFAULT_CURRENCY
    if len(currency) > 8:
        raise ValueError("currency 过长")

    for item in items:
        if not isinstance(item, dict):
            continue
        capability = str(item.get("capability") or "").strip().lower()
        if capability not in CAPABILITIES:
            raise ValueError(f"未知能力类别: {capability or '<empty>'}，可选值: {', '.join(CAPABILITIES)}")
        provider = str(item.get("provider") or "").strip()[:64]
        model = str(item.get("model") or "").strip()[:120]
        price = _require_int(item.get("unit_price_micro"), "unit_price_micro")
        secondary = _require_int(item.get("unit_price_secondary_micro"), "unit_price_secondary_micro")
        multipliers = parse_multipliers(item.get("resolution_multipliers") or {})
        # configured 缺省或显式为 None 时按「是否填了单价」推断，避免前端只改价格
        # 却因为漏传 configured 导致价目被当成未配置。
        raw_configured = item.get("configured")
        configured = bool(raw_configured) if raw_configured is not None else price is not None
        if configured and price is None:
            raise ValueError(f"{capability}/{provider or '*'}/{model or '*'} 标记为已配置时必须填写单价")
        row = (
            db.query(PricingConfig)
            .filter(
                PricingConfig.capability == capability,
                PricingConfig.provider == provider,
                PricingConfig.model == model,
            )
            .first()
        )
        if row is None:
            row = PricingConfig(id=uuid.uuid4().hex, capability=capability, provider=provider, model=model)
            db.add(row)
        row.currency = str(item.get("currency") or currency)[:8] or currency
        row.unit_price_micro = price if configured else None
        row.unit_price_secondary_micro = secondary if configured else None
        row.resolution_multipliers = json.dumps(multipliers, ensure_ascii=False)
        row.configured = configured
        row.note = str(item.get("note") or "")[:200]
    db.commit()
    return list_pricing(db)


__all__ = [
    "COST_SOURCE_LOCAL",
    "COST_SOURCE_PRICING",
    "COST_SOURCE_UNKNOWN",
    "DEFAULT_PRICING_PROVIDERS",
    "LOCAL_ZERO_COST_PROVIDERS",
    "PriceResolution",
    "UNIT_MULTIPLIER_MICRO",
    "apply_multiplier",
    "ceil_div",
    "compute_cost_micro",
    "format_amount_micro",
    "list_pricing",
    "multiplier_for",
    "parse_multipliers",
    "resolve_price",
    "save_pricing",
    "seed_pricing_defaults",
]
