"""任务启动前的 Provider 配置预检。

在 API 入口启动后台任务之前，同步检查本次任务真正需要的模型端点是否已配置；
缺失时立即以稳定错误拒绝启动（error_code=provider_not_configured），而不是让
任务排队后才在执行中途因缺少密钥失败。

检查口径（缺什么才拦什么，能自动回退的能力不拦截）：
- script：LLM 主/备端点至少一个配置了 API Key（与 ``LLMService.available`` 同口径，
  备端点需与主端点不是同一个服务地址）；
- image：永不拦截 —— 未配置密钥时图像服务自动回退占位图（既有约定，见
  ``image_service._resolve_route``）；
- video：视频端点必须配置 API Key（视频没有本地回退）；
- voice：默认必须配置；但当视频适配器声明原生音频能力且已接入厂商实现
  （``audio_routing.native_audio_capable``）时不强制 —— 视频模型能随画面直出
  对白语音时，独立 TTS 不再是必需品。
"""

from __future__ import annotations

from urllib.parse import urlparse

from services.providers.capability_matrix import capability_report
from services.providers.endpoint import endpoint_identity, get_endpoint
from services.providers.registry import UnknownProtocolError, get_adapter

CODE_PROVIDER_NOT_CONFIGURED = "provider_not_configured"

# 支持预检的任务类型（与预算 job_type 口径对齐的子集）。
READY_JOB_TYPES = ("script_pipeline", "shot_video", "shot_audio")

CAPABILITY_LABELS = {
    "script": "剧本解析（LLM）",
    "video": "视频生成模型",
    "voice": "配音（TTS）",
}


class ProviderNotConfiguredError(RuntimeError):
    """预检未通过：携带缺失项列表，由路由层转成结构化 HTTP 响应。"""

    def __init__(self, missing: list[dict]):
        self.missing = missing
        self.message = format_message(missing)
        super().__init__(self.message)


def voice_endpoint_configured() -> bool:
    """语音端点是否配置了 API Key。"""
    return bool(str(get_endpoint("voice").api_key or "").strip())


def native_video_audio_ready(endpoint=None) -> bool:
    """视频端点是否具备真正可用的原生音频能力（能力声明 + 厂商实现已接入）。"""

    from services.audio_routing import native_audio_capable  # 局部导入，避免循环依赖

    return native_audio_capable(endpoint)


def missing_providers(
    job_type: str,
    *,
    mode: str = "",
    has_dialogue: bool | None = None,
    audio_mode_override: str = "",
) -> list[dict]:
    """返回某类任务启动前缺失的 Provider 配置列表；空列表表示可以启动。

    - ``mode``：script_pipeline 的运行模式；auto 模式端到端跑完，还需要视频/配音。
    - ``has_dialogue``：shot_video 用；镜头没有台词时根本不会调用 TTS。
    - ``audio_mode_override``：镜头级音频路径覆盖；显式指定 tts 时必须配置语音端点。
    """

    job_type = str(job_type or "").strip().lower()
    if job_type == "script_pipeline":
        issues = []
        if not _script_ready():
            issues.append(_issue("script"))
        if str(mode or "").strip().lower() == "auto":
            issues.extend(_video_issues())
            voice = _voice_issue(has_dialogue=True, audio_mode_override="")
            if voice:
                issues.append(voice)
        return issues
    if job_type == "shot_video":
        issues = []
        issues.extend(_video_issues())
        voice = _voice_issue(has_dialogue=has_dialogue, audio_mode_override=audio_mode_override)
        if voice:
            issues.append(voice)
        return issues
    if job_type == "shot_audio":
        # 纯配音任务本身就是 TTS 调用，原生音频能力替代不了它。
        voice = _voice_issue(has_dialogue=True, audio_mode_override="tts")
        return [voice] if voice else []
    raise ValueError(f"未知任务类型: {job_type or '<empty>'}，可选值: {', '.join(READY_JOB_TYPES)}")


def ensure_task_providers_ready(job_type: str, **hints) -> None:
    """预检不通过时抛出 ``ProviderNotConfiguredError``。"""

    missing = missing_providers(job_type, **hints)
    if missing:
        raise ProviderNotConfiguredError(missing)


def format_message(missing: list[dict]) -> str:
    names = "、".join(str(item.get("label") or item.get("capability") or "") for item in missing)
    return f"以下模型端点尚未配置 API Key：{names}。请在「系统设置 → 模型服务」补齐后重试。"


# ---------------------------------------------------------------------------
# 各能力的配置判定
# ---------------------------------------------------------------------------


def _script_ready() -> bool:
    endpoint = get_endpoint("script")
    if str(endpoint.api_key or "").strip():
        return True
    # 备端点：配置了密钥、且与主端点不是同一个服务地址时可用（与 LLMService 口径一致）。
    fallback = get_endpoint("script_fallback")
    return bool(str(fallback.api_key or "").strip()) and endpoint_identity(fallback.base_url) != endpoint_identity(
        endpoint.base_url
    )


def _video_ready() -> bool:
    endpoint = get_endpoint("video")
    if not str(endpoint.api_key or "").strip():
        return False
    return not _video_preflight_issues()


def _video_issues() -> list[dict]:
    """视频端点的启动前问题列表（API key + 预检明细）。"""
    endpoint = get_endpoint("video")
    if not str(endpoint.api_key or "").strip():
        return [_issue("video")]
    detail = _video_preflight_issues()
    if not detail:
        return []
    return [
        {
            "capability": "video",
            "label": CAPABILITY_LABELS.get("video", "video"),
            "message": "视频 Provider 预检未通过: " + "；".join(detail),
        }
    ]


def video_provider_preflight() -> dict:
    """视频 Provider 启动前预检报告（不含任何密钥内容）。

    验证 endpoint、model、API key、首帧参考图能力与固定时长能力；
    本流水线要求视频 Provider 支持 ``reference_image``（first_frame_only 的
    图生视频），不支持时预检失败并给出明确原因。
    """
    endpoint = get_endpoint("video")
    report: dict = {
        "capability": "video",
        "protocol": endpoint.protocol,
        "base_url": endpoint.base_url,
        "base_url_host": urlparse(endpoint.base_url or "").hostname or "",
        "model": endpoint.model,
        "api_key_configured": bool(str(endpoint.api_key or "").strip()),
        "adapter_registered": True,
        "reference_image": False,
        "reference_mode": "text_only",
        "native_audio": False,
        "fixed_duration": None,
        "min_duration": None,
        "max_duration": None,
        "duration_step": None,
        "provider_capabilities": {},
        "control_types_supported": [],
        "reference_weight_policy": "text_only_policy",
        "issues": [],
    }
    try:
        adapter_cls = get_adapter("video", endpoint.protocol)
    except UnknownProtocolError as exc:
        report["adapter_registered"] = False
        report["issues"].append(f"视频协议未注册适配器: {exc}")
        return report

    capabilities = adapter_cls.effective_capabilities(endpoint.model)
    matrix = capability_report("video", endpoint.protocol, model=endpoint.model, adapter_cls=adapter_cls)
    report["provider_capabilities"] = matrix
    report["control_types_supported"] = matrix.get("supported", [])
    report["reference_weight_policy"] = matrix.get("reference_weight_policy", "text_only_policy")
    if capabilities is not None:
        report["reference_image"] = bool(capabilities.reference_image)
        report["reference_mode"] = str(getattr(capabilities, "reference_mode", "text_only"))
        report["native_audio"] = bool(capabilities.native_audio)
        report["fixed_duration"] = getattr(capabilities, "fixed_duration", None)
        report["min_duration"] = getattr(capabilities, "min_duration", None)
        report["max_duration"] = getattr(capabilities, "max_duration", None)
        report["duration_step"] = getattr(capabilities, "duration_step", None)
    if not report["api_key_configured"]:
        report["issues"].append("视频端点未配置 API Key（视频没有本地回退，任务会被拒绝启动）")
    report["issues"].extend(_video_preflight_issues())
    return report


def _video_preflight_issues() -> list[str]:
    """视频端点的配置级问题（不触网、不读取密钥值）。"""
    endpoint = get_endpoint("video")
    issues: list[str] = []
    if not str(endpoint.protocol or "").strip():
        issues.append("视频协议为空")
        return issues
    try:
        adapter_cls = get_adapter("video", endpoint.protocol)
    except UnknownProtocolError as exc:
        issues.append(f"视频协议未注册适配器: {exc}")
        return issues
    if not str(endpoint.model or "").strip():
        issues.append(
            f"视频协议 {endpoint.protocol} 未配置模型名（model），请在「系统设置 → 模型服务」选择账号实际可用的模型"
        )
    # 模型级能力判断优先；测试替身/未声明 effective_capabilities 的适配器回落类级声明。
    effective = getattr(adapter_cls, "effective_capabilities", None)
    capabilities = effective(endpoint.model) if callable(effective) else getattr(adapter_cls, "capabilities", None)
    if capabilities is not None and not capabilities.reference_image:
        issues.append(
            f"视频协议 {endpoint.protocol} 不支持首帧参考图（first_frame_only 图生视频），"
            "无法满足「已审核故事板首帧驱动」的生成流程"
        )
    return issues


def _voice_issue(*, has_dialogue: bool | None, audio_mode_override: str) -> dict | None:
    if has_dialogue is False:
        return None  # 没有台词的镜头不会调用 TTS
    override = str(audio_mode_override or "").strip().lower()
    if override == "native":
        return None
    if override != "tts" and native_video_audio_ready():
        # 视频模型支持原生对白语音：不强制要求 TTS。
        return None
    if voice_endpoint_configured():
        return None
    return _issue("voice")


def _issue(capability: str) -> dict:
    return {
        "capability": capability,
        "label": CAPABILITY_LABELS.get(capability, capability),
        "message": f"{CAPABILITY_LABELS.get(capability, capability)}未配置 API Key",
    }


__all__ = [
    "CODE_PROVIDER_NOT_CONFIGURED",
    "READY_JOB_TYPES",
    "ProviderNotConfiguredError",
    "ensure_task_providers_ready",
    "format_message",
    "missing_providers",
    "native_video_audio_ready",
    "video_provider_preflight",
    "voice_endpoint_configured",
]
