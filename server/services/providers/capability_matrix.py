"""Provider Capability Matrix 的唯一声明源。

这里把适配器的真实协议能力转换成对外稳定的能力状态，供 Prompt、API DTO、日志、
任务报告和前端共同消费。状态只允许：

- ``supported``：已接入并且请求载荷确实能携带该能力；
- ``partial``：仅特定模型/协议变体或有限参数可用；
- ``unsupported``：当前 Provider/适配器不支持，禁止在文案中暗示已生效。

参考权重单独使用 ``policy`` 字段：Provider 没有数值权重参数时必须是
``text_only_policy``，不得把提示词里的偏好伪装成模型权重。
"""

from __future__ import annotations

from typing import Any

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"

TEXT_ONLY_POLICY = "text_only_policy"


class CapabilityDowngradeRequiredError(RuntimeError):
    """需要人工确认能力降级，未确认时必须阻止生成。"""

    code = "capability_downgrade_confirmation_required"

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}

FEATURE_ORDER = (
    "multiple_reference_images",
    "reference_weights",
    "character_identity",
    "scene_reference",
    "first_frame",
    "last_frame",
    "first_last_frame_interpolation",
    "pose_control",
    "depth_control",
    "lora",
    "ip_adapter",
    "native_audio",
    "voice_consistent",
    "dialogue_in_prompt",
)

# --- 视频逐镜头路由（模型级能力判断） ----------------------------------------
#
# 镜头级视频生成路由模式，按镜头要求与 ``effective_capabilities(model)`` 共同裁决：
# - ``first_frame_i2v``：图生视频，仅已审核故事板首帧驱动；
# - ``multi_reference_r2v``：参考生视频，首帧 + 角色身份/场景基准/连续性多参考；
# - ``first_last_frame``：首尾帧插值，要求模型声明 last_frame_input。

VIDEO_MODE_FIRST_FRAME_I2V = "first_frame_i2v"
VIDEO_MODE_MULTI_REFERENCE_R2V = "multi_reference_r2v"
VIDEO_MODE_FIRST_LAST_FRAME = "first_last_frame"
VIDEO_MODES = (
    VIDEO_MODE_FIRST_FRAME_I2V,
    VIDEO_MODE_MULTI_REFERENCE_R2V,
    VIDEO_MODE_FIRST_LAST_FRAME,
)

# r2v 多参考素材的发送优先级：首帧锚定画面，角色身份一致性最高，其次场景
# 基准，最后连续性参考；超出数量上限时从优先级最低的一端丢弃。
REFERENCE_SEND_PRIORITY = (
    "approved_storyboard_first_frame",
    "end_frame",
    "character_three_view",
    "scene_baseline",
    "continuity_frame",
)

_REFERENCE_PRIORITY_UNKNOWN = len(REFERENCE_SEND_PRIORITY)


def reference_send_priority(asset_type: str) -> int:
    """参考素材的发送优先级序号；越小越先发送，未知类型排在已知类型之后。"""

    try:
        return REFERENCE_SEND_PRIORITY.index(str(asset_type or ""))
    except ValueError:
        return _REFERENCE_PRIORITY_UNKNOWN


def select_video_mode(required_capabilities: Any, capabilities: Any) -> str:
    """按镜头要求与模型生效能力选择视频路由模式。

    只读 ``effective_capabilities(model)`` 的结果：首尾帧插值要求最具体、优先
    满足；其次多参考 r2v；模型能力不满足时回落 ``first_frame_i2v``，是否接受
    该降级由调用方的能力门禁决定，这里不做静默替换。
    """

    required = {str(item) for item in (required_capabilities or [])}
    if (
        required & {"last_frame", "first_last_frame_interpolation"}
        and _value(capabilities, "last_frame_input", False)
        and _value(capabilities, "first_last_frame_interpolation", False)
    ):
        return VIDEO_MODE_FIRST_LAST_FRAME
    if "multiple_reference_images" in required and _value(capabilities, "multiple_reference_images", False):
        return VIDEO_MODE_MULTI_REFERENCE_R2V
    return VIDEO_MODE_FIRST_FRAME_I2V


def supports_first_last_frame(capabilities: Any) -> bool:
    """只有协议同时明确声明首帧、尾帧输入和首尾帧插值时才允许 end_frame。"""

    return bool(
        (_value(capabilities, "reference_image", False) or _value(capabilities, "first_frame", False))
        and _value(capabilities, "last_frame_input", False)
        and _value(capabilities, "first_last_frame_interpolation", False)
    )


def feature(status: str, **detail: Any) -> dict[str, Any]:
    """构造一个能力项；detail 只放可验证的协议事实，不放产品承诺。"""

    return {"status": status if status in {SUPPORTED, PARTIAL, UNSUPPORTED} else UNSUPPORTED, **detail}


def _value(capabilities: Any, name: str, default: Any = None) -> Any:
    return getattr(capabilities, name, default)


def _effective_capabilities(adapter_cls: Any, model: str = "") -> Any:
    capabilities = getattr(adapter_cls, "capabilities", None)
    effective = getattr(adapter_cls, "effective_capabilities", None)
    if callable(effective):
        try:
            return effective(model)
        except TypeError:
            # 实例方法/测试适配器的兼容回退。
            try:
                return effective(None, model)
            except Exception:
                return capabilities
        except Exception:
            return capabilities
    return capabilities


def capability_report(
    capability: str,
    protocol: str,
    *,
    model: str = "",
    adapter_cls: Any = None,
    production_ready: bool | None = None,
) -> dict[str, Any]:
    """返回指定 Provider 的 Capability Matrix DTO。"""

    capability = str(capability or "").strip().lower()
    protocol = str(protocol or "").strip().lower()
    if adapter_cls is None:
        from services.providers.registry import UnknownProtocolError, get_adapter

        try:
            adapter_cls = get_adapter(capability, protocol)
        except UnknownProtocolError:
            return {
                "capability": capability,
                "provider": protocol,
                "model": model,
                "adapter_registered": False,
                "production_ready": False,
                "reference_mode": "unsupported",
                "features": {name: feature(UNSUPPORTED) for name in FEATURE_ORDER},
                "reference_weight_policy": TEXT_ONLY_POLICY,
            }

    caps = _effective_capabilities(adapter_cls, model)
    ready = getattr(adapter_cls, "production_ready", True) if production_ready is None else production_ready
    ref_images = bool(_value(caps, "reference_images", False) or _value(caps, "reference_image", False))
    ref_mode = str(_value(caps, "reference_mode", "text_only") or "text_only")
    multiple = bool(_value(caps, "multiple_reference_images", _value(caps, "reference_images", False)))
    if ref_mode in {"multi_reference", "model_conditional"}:
        multiple = True
    max_refs = _value(caps, "max_reference_images", _value(caps, "max_reference_image_count", 0)) or 0
    weight_parameter = str(_value(caps, "reference_weight_parameter", "") or "")
    weight_policy = str(_value(caps, "reference_weight_policy", TEXT_ONLY_POLICY) or TEXT_ONLY_POLICY)
    role_parameter = str(_value(caps, "reference_role_parameter", "") or "")
    first_frame = bool(_value(caps, "reference_image", False) or _value(caps, "first_frame", False))
    last_frame = bool(_value(caps, "last_frame_input", False))
    interpolation = bool(first_frame and last_frame and _value(caps, "first_last_frame_interpolation", False))
    character_param = str(_value(caps, "character_reference_parameter", "") or "")
    scene_param = str(_value(caps, "scene_reference_parameter", "") or "")

    features: dict[str, dict[str, Any]] = {
        "multiple_reference_images": feature(
            SUPPORTED if multiple and ref_mode != "model_conditional" else PARTIAL if multiple else UNSUPPORTED,
            parameter=str(_value(caps, "reference_parameter", "image") or "image"),
            max_count=int(max_refs) if max_refs else None,
            role_parameter=role_parameter or None,
            condition="model must contain -r2v" if ref_mode == "model_conditional" else None,
        ),
        "reference_weights": feature(
            SUPPORTED if weight_parameter else UNSUPPORTED,
            parameter=weight_parameter or None,
            policy=weight_policy if not weight_parameter else None,
            note=None if weight_parameter else "数值权重未接入；只记录 text_only_policy，不发送伪权重",
        ),
        "character_identity": feature(
            SUPPORTED if character_param else PARTIAL if multiple else UNSUPPORTED,
            parameter=character_param or role_parameter or None,
        ),
        "scene_reference": feature(
            SUPPORTED if scene_param else PARTIAL if multiple else UNSUPPORTED,
            parameter=scene_param or role_parameter or None,
        ),
        "first_frame": feature(SUPPORTED if first_frame else UNSUPPORTED, parameter="reference_image" if first_frame else None),
        "last_frame": feature(SUPPORTED if last_frame else UNSUPPORTED, parameter="last_frame" if last_frame else None),
        "first_last_frame_interpolation": feature(SUPPORTED if interpolation else UNSUPPORTED),
        "pose_control": feature(SUPPORTED if _value(caps, "openpose", False) else UNSUPPORTED),
        "depth_control": feature(SUPPORTED if _value(caps, "depth", False) else UNSUPPORTED),
        "lora": feature(SUPPORTED if _value(caps, "lora", False) else UNSUPPORTED),
        "ip_adapter": feature(SUPPORTED if _value(caps, "ip_adapter", False) else UNSUPPORTED),
        "native_audio": feature(SUPPORTED if _value(caps, "native_audio", False) and ready else UNSUPPORTED),
        "voice_consistent": feature(SUPPORTED if _value(caps, "voice_consistent", False) and ready else UNSUPPORTED),
        "dialogue_in_prompt": feature(SUPPORTED if _value(caps, "dialogue_in_prompt", False) and ready else UNSUPPORTED),
    }

    return {
        "capability": capability,
        "provider": protocol,
        "model": model,
        "adapter_registered": True,
        "production_ready": bool(ready),
        "reference_mode": ref_mode if ref_images else "text_only",
        "reference_weight_policy": weight_policy if not weight_parameter else "provider_parameter",
        "reference_role_parameter": role_parameter or None,
        "features": features,
        "supported": [name for name in FEATURE_ORDER if features[name]["status"] == SUPPORTED],
        "partial": [name for name in FEATURE_ORDER if features[name]["status"] == PARTIAL],
        "unsupported": [name for name in FEATURE_ORDER if features[name]["status"] == UNSUPPORTED],
    }


def status_of(report: dict[str, Any], name: str) -> str:
    return str(((report or {}).get("features") or {}).get(name, {}).get("status") or UNSUPPORTED)


def payload_control_types(
    capability: str,
    report: dict[str, Any],
    *,
    has_first_frame: bool = False,
    has_end_frame: bool = False,
    has_pose_control: bool = False,
    has_depth_control: bool = False,
) -> list[str]:
    """只返回确实能出现在 Provider 请求载荷里的控制类型。"""

    sent: list[str] = []
    if capability == "image" and status_of(report, "multiple_reference_images") in {SUPPORTED, PARTIAL}:
        sent.append("reference_images")
    if capability == "video" and has_first_frame and status_of(report, "first_frame") == SUPPORTED:
        sent.append("first_frame")
    if capability == "video" and status_of(report, "multiple_reference_images") in {SUPPORTED, PARTIAL}:
        sent.append("reference_images")
    if status_of(report, "reference_weights") == SUPPORTED:
        sent.append("reference_weights")
    if has_pose_control and status_of(report, "pose_control") == SUPPORTED:
        sent.append("openpose")
    if has_depth_control and status_of(report, "depth_control") == SUPPORTED:
        sent.append("depth")
    for name, control in (("lora", "lora"), ("ip_adapter", "ip_adapter")):
        if status_of(report, name) == SUPPORTED:
            sent.append(control)
    if has_end_frame and status_of(report, "first_last_frame_interpolation") == SUPPORTED:
        sent.append("first_last_frame_interpolation")
    return list(dict.fromkeys(sent))
