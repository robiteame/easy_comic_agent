# 视频模型原生 API 适配

依据《视频生成模型 API 调用方式差异调研（2026-11 版）》实现。每家使用独立协议，统一入口仍为 `VideoService` / `VideoRequest`。在「系统设置 → 模型与 API」选择厂商预设，填写密钥和账号实际可用的模型名称并保存；不会自动修改现有配置。

## 已实现的协议

| 厂商 | Protocol | 已实现的主要差异 |
| --- | --- | --- |
| OpenAI Sora | `openai-sora` | `seconds` 字符串枚举、`size` 像素尺寸、单首帧、鉴权 `/content` 下载 |
| Google Vertex Veo | `google-veo` | `predictLongRunning` / `fetchPredictOperation`、4/6/8 秒、首尾帧、3.1 最多 3 张参考图、Base64 / GCS 结果 |
| Runway | `runway` | 版本头、像素对比例、Gen-4.5 的 2–10 秒、Turbo 的 5/10 秒、自动上传本地首帧 |
| 可灵 Kling | `kling` | AK/SK 签 JWT、`model_name`、字符串时长、`image` / `image_tail`、按模型选择时长与音频能力 |
| MiniMax H3 / H3-Max | `minimax` | `content[]` + `role`、图生自适应比例、版本分辨率、图/视频/音频数量限制 |
| Vidu Q3 | `vidu` | Token 鉴权、文生/图生/首尾帧/参考图独立端点、1–16 秒、不发送比例字段 |
| 火山 Seedance | `ark-seedance` | 保留旧代协议；2.x/2.5 多模态参考、首尾帧、模型级时长、显式编辑/延长任务 |
| 阿里 Wan | `dashscope-wanx` | 保留旧代协议；2.7 i2v 的 `media[]`、`last_frame`、驱动音频、视频续写、2–15 秒 |
| PixVerse | `pixverse` | 图片上传换 ID、每次请求的 trace UUID、生成端点分流、业务错误及视频状态查询 |
| Luma | `luma` | `keyframes.frame0/frame1`、`5s` / `9s`、图片或历史 generation ID 关键帧 |

所有新协议均完成创建任务、轮询、失败处理和结果落盘。结果链接立即下载到项目目录，继续沿用存储配额、媒体大小限制和 FFmpeg 尾帧提取。创建计费任务的请求只提交一次，避免响应丢失后自动重试产生重复费用；查询请求保留瞬态重试。

## 配置与素材

设置页提供十家预设。Base URL 可填写官方版本根地址或同协议网关地址。官方协议和云平台的托管版协议不能混用；本次没有把百炼/腾讯的 Kling、PixVerse 包装成官方请求体。

- **Kling**：密钥格式 `AccessKey:SecretKey`，每次请求自动签发短期 JWT。4.0 的确切模型 ID 未在研究文档确认，预设使用明确的 2.6；可手动填写控制台模型 ID。可通过 `params.mode`、`cfg_scale`、`camera_control` 等配置已知扩展字段。
- **Vertex Veo**：密钥字段填写有效的 Google OAuth bearer token；设置页填写项目 ID 和 Location。也可把 Base URL 配成完整的模型资源路径。GCS 输出需要同一令牌具备对象读取权限。令牌过期后需要更新。当前生成单条结果，`sample_count > 1` 会拒绝，避免多条计费结果被丢弃。
- **Sora**：基础版使用 720p、16:9 或 9:16。Pro 可用更高尺寸，也可配置 `params.size`。本地/内联首帧会裁切为目标精确尺寸后上传；公网 URL 首帧须由调用方保证尺寸匹配。
- **Runway / PixVerse**：本地和内联首帧由适配器走厂商上传接口；Runway 也接受已有 `runway://` URI。
- **MiniMax / Luma**：使用公网图片 URL。若服务输出目录已经有公网托管地址，可在设置页的「公网输出目录地址」填写，例如 `https://media.example.com/output`。仅将 `OUTPUT_DIR` 内已有文件映射为对应 URL，不会替用户开通托管或上传。Luma 可用 `generation:<id>` 表示历史任务关键帧。
- **MiniMax H3**：统一 720p 映射为 `768P`，1080p 映射为 `2K`；H3-Max 仅允许 `480P` / `768P`，可用 `params.resolution` 明确配置。
- **音频**：沿用 `audio_mode=tts/native/auto` 路由。有开关的厂商按所选路径发送开关；H3、Wan 2.7 无文档确认的关闭开关，结果如实标记自带音轨。原生音频结果继续用 ffprobe 校验。

通过配置文件或 API 可在视频端点的 `params` 中设置 `max_poll_attempts`、`poll_interval_seconds`。不要将密钥放入 `params`，这里的参数会回显给设置页。

## 请求与能力约束

统一 `VideoRequest.duration` 使用整数秒，适配器转换为厂商格式。时长枚举也进入镜头规划：例如 Luma 的 6 秒叙事镜头会生成合法的 9 秒片段，再按执行计划剪辑。Seedance 编辑任务通过 `task_type="edit"` 转换为服务端要求的 `duration=-1`。

统一请求增加 `negative_prompt`、`generate_audio`、`reference_video`、`reference_audio`、`task_type`。多素材继续通过 `reference_assets` 表达；`provider_type` 可指定 `reference_image` / `reference_video` / `reference_audio`。已审核首帧使用 `reference_image`，尾帧使用 `end_frame`。不支持的素材或任务组合会在提交前报错。

以下分支仍明确拒绝，不能视为已接入：

- Gemini Omni 的 `generateContent` 视频结构、Runway Aleph 编辑的完整请求结构，研究文档未给出可执行的 REST 契约。
- Kling 多模态参考端点与 4.0 未公开的字段组合。
- MiniMax 旧 Hailuo 2.3（不能把 H3 `content[]` 套用到旧接口）。
- PixVerse 任意参考视频、Vidu 视频/音频参考、Luma 任意视频 URL 输入。
- Sora Azure 插值、角色库、remix、extensions 等独立接口。

多数原生厂商没有已接入的统一模型列表或免费连接探测；设置页会明确提示手动填写模型名/无法免费探测，不会通过创建付费视频来测试连接。

## 研究文档的已核对修正

PixVerse 官方请求形状与研究文档简写不同，代码按已核对的官方接口实现：

- 首尾帧使用 `first_frame_img` / `last_frame_img`，值为上传后的图片 ID：[官方说明](https://docs.platform.pixverse.ai/first-and-last-frame-video-43051213e0.md)。
- 多参考图使用 `/openapi/v2/video/fusion/generate` 和 `image_references`。
- 查询使用 `/openapi/v2/video/result/{video_id}`，可通过 `params.query_path` 覆盖。
- Runway 自动上传使用 `uploadUrl` / `fields` / `runwayUri`：[官方参考](https://docs.dev.runwayml.com/api/#tag/Uploads)。

## 验证

协议测试使用 HTTP mock，不消费厂商额度。覆盖字段/类型映射、JWT、上传、提交与轮询、异步失败、下载配额、配置隔离、模型时长枚举和现有视频流程回归。

```sh
server/.venv/bin/python -m pytest server/tests -q
pnpm --dir client run typecheck
```

未进行真实账号的付费生成；实际模型权限、未确认型号及线上参数范围仍需使用对应账号验证。
