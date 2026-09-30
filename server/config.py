from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 项目路径
    BASE_DIR: Path = Path(__file__).parent.parent
    SERVER_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = Path(__file__).parent / "data"
    OUTPUT_DIR: Path = Path(__file__).parent.parent / "output"
    ASSETS_DIR: Path = Path(__file__).parent.parent / "assets"

    # LLM 配置
    LLM_PROVIDER: str = "openai"  # openai / deepseek / mimo / seeddance
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    LLM_MAX_TOKENS: int = 4096
    # 任务失败原因的 LLM 自动识别：关闭后只保留规则分类（离线/测试环境用）。
    ERROR_ANALYSIS_LLM_ENABLED: bool = True

    # Mimo (小米 MiMo, 通过硅基流动 SiliconFlow 调用)
    MIMO_API_KEY: str = ""
    MIMO_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
    MIMO_MODEL: str = "mimo-v2.5"
    MIMO_MULTIMODAL_MODEL: str = "mimo-v2-omni"
    MIMO_TTS_MODEL: str = "mimo-v2.5-tts"
    MIMO_TTS_VOICE: str = "冰糖"
    MIMO_TTS_FORMAT: str = "wav"

    # SeedDance / Seedream (字节跳动, 通过火山引擎方舟平台调用)
    ARK_API_KEY: str = ""
    SEEDDANCE_API_KEY: str = ""
    SEEDREAM_API_KEY: str = ""
    SEEDDANCE_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    SEEDDANCE_MODEL: str = "doubao-seedance-2-0-260128"
    VIDEO_PROVIDER: str = "Doubao-Seedance-2.0"
    SEEDREAM_MODEL: str = "doubao-seedream-5-0-lite-260128"
    SEEDREAM_IMAGE_SIZE: str = "1440x2560"

    # 阿里云百炼 DashScope (通义万相视频; VIDEO_PROVIDER 填 wanx/dashscope 时启用)
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/api/v1"
    DASHSCOPE_VIDEO_MODEL: str = "wan2.7-i2v"

    # 阿里云百炼 Qwen-Image（IMAGE_PROVIDER=qwen-image 时启用）
    QWEN_IMAGE_API_KEY: str = ""
    QWEN_IMAGE_BASE_URL: str = "https://dashscope.aliyuncs.com/api/v1"
    QWEN_IMAGE_MODEL: str = "qwen-image-3.0"
    QWEN_IMAGE_SIZE: str = "1440x2560"

    # 阿里云百炼语音合成 CosyVoice（TTS_PROVIDER 填 dashscope/bailian/cosyvoice 时启用）
    DASHSCOPE_TTS_API_KEY: str = ""
    DASHSCOPE_TTS_BASE_URL: str = "https://dashscope.aliyuncs.com/api/v1"
    DASHSCOPE_TTS_MODEL: str = "cosyvoice-v2"
    DASHSCOPE_TTS_VOICE: str = "longwan_v2"
    DASHSCOPE_TTS_FORMAT: str = "wav"

    # 腾讯云语音合成 (配置 SecretId/Key 后语音端点默认切换到 tencent-tts 协议)
    TENCENT_SECRET_ID: str = ""
    TENCENT_SECRET_KEY: str = ""
    TENCENT_TTS_URL: str = "https://tts.tencentcloudapi.com"
    TENCENT_TTS_VOICE: str = "101001"
    TENCENT_TTS_FORMAT: str = "wav"

    # 图像生成配置
    # local = 无密钥占位图 stub(PIL 生成,使全流程可离线跑通); stability / ark-seedream / qwen-image = 真实云端服务
    # 配置真实 provider 但缺少对应 API Key 时,会自动回退到占位图,不再报错中断
    IMAGE_PROVIDER: str = "local"
    STABILITY_API_KEY: str = ""
    STABILITY_API_URL: str = "https://api.stability.ai/v2beta"
    SD_LOCAL_URL: str = "http://127.0.0.1:7860"

    # TTS 配置
    TTS_PROVIDER: str = "mimo"  # 语音协议选择：mimo / tencent / dashscope（bailian、cosyvoice 等别名亦可）
    TTS_DEFAULT_VOICE: str = "zh-CN-XiaoyiNeural"

    # ChromaDB 配置
    CHROMADB_PATH: str = str(Path(__file__).parent / "data" / "chromadb")

    # 数据库配置
    DATABASE_URL: str = f"sqlite:///{Path(__file__).parent / 'data' / 'comic_agent.db'}"

    # LangGraph 配置
    CHECKPOINT_PATH: str = str(Path(__file__).parent / "data" / "checkpoints")

    # 渲染配置
    DEFAULT_FPS: int = 24
    DEFAULT_RESOLUTION: str = "1080p"
    # 请求与媒体资源限制。限制放在配置中，便于桌面版/服务端按机器容量调整。
    MAX_SCRIPT_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_VIDEO_UPLOAD_BYTES: int = 1024 * 1024 * 1024
    MAX_SCRIPT_TEXT_CHARS: int = 1_000_000
    MAX_REMOTE_MEDIA_BYTES: int = 512 * 1024 * 1024
    MAX_IMAGE_GENERATION_BYTES: int = 32 * 1024 * 1024
    MAX_INLINE_REFERENCE_BYTES: int = 12 * 1024 * 1024
    # 参考图内联（base64）的全局兜底预算（编码后字节数）。协议有已知的更高
    # 网关上限时由适配器 capabilities.max_reference_inline_bytes 覆盖
    # （如 ark-seedance 8MB）；未声明的协议沿用此值，超预算图片走 JPEG
    # 压缩阶梯降级，压不进预算则明确报错而不是把超限请求发出去。
    VIDEO_REFERENCE_INLINE_BUDGET_BYTES: int = 128 * 1024
    # 需要参考图的阶段（角色三视图 / 场景基准图 / 绑定资产的定稿故事板）在
    # 「当前图像 Provider 声明不支持参考图」且「没有其它已配置的参考图 Provider」
    # 时的行为：
    #   prefer（默认）—— 明确告警并如实记录 references_sent=0 后继续生成，
    #                    绝不假装参考图已生效；界面与日志都会标注能力限制。
    #   strict        —— 直接阻止生成，要求先切换到支持参考图的 Provider。
    # 两种取值都不会静默丢弃参考图。
    IMAGE_REFERENCE_ENFORCEMENT: str = "prefer"
    # 视觉一致性验证：vlm=调用已配置的视觉模型复核生成结果；off=只记录载荷指标。
    CONSISTENCY_VALIDATION_MODE: str = "vlm"
    # provider 出站请求是否继承系统/环境代理（httpx trust_env）。
    # 全部端点都在国内直连可达的部署建议在 .env 里关闭，避免本机代理
    # 把 API 请求绕到不稳定链路上。
    PROVIDER_HTTP_TRUST_ENV: bool = True
    MAX_TTS_AUDIO_BYTES: int = 64 * 1024 * 1024
    # 字幕与音频混音工作台的上限（同样属于服务端安全边界）。
    MAX_AUDIO_UPLOAD_BYTES: int = 128 * 1024 * 1024
    MAX_AUDIO_TRACKS: int = 32
    MAX_SUBTITLE_TRACKS: int = 8
    MAX_SUBTITLE_CUES: int = 2000
    MAX_SUBTITLE_CUE_CHARS: int = 500
    MAX_SUBTITLE_IMPORT_CHARS: int = 2 * 1024 * 1024
    MAX_SUBTITLE_CHARACTER_CHARS: int = 60
    # 混音输出与预览共用的响度/削波目标（EBU R128 短视频常见档位）。
    LOUDNESS_TARGET_I: float = -16.0
    LOUDNESS_TARGET_TP: float = -1.5
    LOUDNESS_TARGET_LRA: float = 11.0
    CLIPPING_HEADROOM_DB: float = 0.1
    PROJECT_STORAGE_QUOTA_BYTES: int = 5 * 1024 * 1024 * 1024
    PROJECT_TEMP_FILE_TTL_SECONDS: int = 24 * 60 * 60
    PROJECT_VERSION_RETENTION_COUNT: int = 3
    FFMPEG_WORKSPACE_RESERVE_BYTES: int = 1024 * 1024 * 1024
    FFMPEG_TIMEOUT_SECONDS: int = 900
    BACKGROUND_TASK_CANCEL_TIMEOUT_SECONDS: int = 30

    # 任务中心：分页、事件快照与历史清理的上限。列表读取永远分页，
    # 避免任务很多时把整张表读进内存；事件快照只带最近的一批任务。
    JOB_LIST_DEFAULT_PAGE_SIZE: int = 20
    JOB_LIST_MAX_PAGE_SIZE: int = 100
    JOB_EVENT_SNAPSHOT_LIMIT: int = 100
    JOB_CLEANUP_MAX_ROWS: int = 500
    # 只有同类型任务攒够这么多次真实耗时样本，才给出预计剩余时间。
    JOB_ETA_MIN_SAMPLES: int = 3

    # 请求 DTO 与 LLM 输出的字段级上限。前端 min/max 只是交互提示，
    # 真正的边界统一读这里，保证服务端校验与生成流程使用同一套配额。
    MAX_PROJECT_TITLE_CHARS: int = 120
    MAX_PROJECT_GENRE_CHARS: int = 40
    MAX_PROJECT_STYLE_CHARS: int = 48
    MAX_GENERATION_PROMPT_CHARS: int = 4000
    MAX_SHOT_TEXT_CHARS: int = 2000
    MAX_VISUAL_NOTES_CHARS: int = 2000
    MAX_BATCH_SHOT_IDS: int = 200
    MAX_CHARACTER_ASSET_IDS: int = 50
    MIN_TARGET_DURATION_SECONDS: int = 5
    MAX_TARGET_DURATION_SECONDS: int = 600
    MIN_SHOT_DURATION_SECONDS: float = 0.5
    MAX_SHOT_DURATION_SECONDS: float = 60.0
    MAX_EPISODE_NUMBER: int = 9999
    LLM_MAX_CHARACTERS: int = 6
    LLM_MAX_SCENES: int = 8
    LLM_MAX_SHOTS: int = 12
    LLM_MAX_DIALOGUE_LINES: int = 20
    LLM_MAX_TEXT_CHARS: int = 2000
    LLM_MAX_PROMPT_CHARS: int = 2000

    class Config:
        env_file = (
            str(Path(__file__).parent / ".env"),
            str(Path(__file__).parent.parent / ".env"),
        )
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
