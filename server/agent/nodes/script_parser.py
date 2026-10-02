import json
import logging
import re

from agent.output_schemas import (
    CharacterOutput,
    LLMOutputError,
    SceneOutput,
    ScriptParseOutput,
    parse_script_output,
)
from agent.state import AgentState
from config import settings
from memory.project_memory import ProjectMemory
from rag.rag_service import RAGService
from services.llm_service import LLMOutputTruncatedError, LLMService, large_json_max_tokens
from services.prompts import (
    SCRIPT_PARSE_JSON_CONTRACT,
    SCRIPT_PARSE_SYSTEM_PROMPT,
    resolve_json_system_prompt,
)
from services.tts_service import normalize_mimo_voice

llm_service = LLMService()
rag_service = RAGService()
project_memory = ProjectMemory()

logger = logging.getLogger(__name__)

DEFAULT_EMOTION_VARIANTS = {
    "neutral": "calm expression",
    "happy": "gentle smile",
    "shy": "slight blush",
    "sad": "downcast eyes",
    "angry": "determined eyes",
    "surprised": "wide eyes",
}

# 场次/章节自然边界：第X场/第X幕/第X章/场景X/SCENE X/Markdown 标题行。
_SCENE_BOUNDARY = re.compile(
    r"^\s*(?:第\s*[\d一二两三四五六七八九十百千]+\s*[场幕章回集话](?:之[\d一二两三四五六七八九十]+)?"
    r"|\d+[、.．]\s*\S"
    r"|场\s*景\s*[\d一二两三四五六七八九十]+"
    r"|SCENE\s+\d+"
    r"|#{1,3}\s+\S).*$",
    re.IGNORECASE,
)


async def run(state: AgentState) -> dict:
    """Parse free-form input into characters and script scenes."""
    if state.get("storyboard_confirmed") and state.get("characters"):
        return {"current_step": "parse_script"}

    project_id = state["project_id"]
    user_input = (state.get("user_input") or "").strip()
    prompt_revisions = [str(item.get("instruction") or item) for item in state.get("prompt_revisions") or [] if item]
    prompt_revision = str(state.get("prompt_revision") or "").strip()
    if prompt_revision:
        prompt_revisions.append(prompt_revision)
    if prompt_revisions:
        user_input = user_input + "\n\n自动 Critic 修订要求：\n- " + "\n- ".join(prompt_revisions[-3:])
    rag_context: list[str] = []

    if state.get("input_type") == "file" and state.get("uploaded_file_path"):
        try:
            await rag_service.ingest_document(
                project_id=project_id,
                file_path=state["uploaded_file_path"],
                file_type=state.get("file_type", "txt"),
                doc_type="script",
            )
            rag_context = await rag_service.query_context(
                project_id=project_id,
                query="主要人物、场景、冲突、对白",
                n_results=8,
            )
        except Exception:
            rag_context = []

    if not llm_service.available:
        raise RuntimeError("未配置可用的 Mimo/LLM API Key，无法解析剧本")
    if not user_input:
        raise RuntimeError("剧本内容为空，无法解析")

    effective_style = state.get("effective_style") or state.get("style") or "anime"
    prefer_fallback = str((state.get("provider_switch") or {}).get("director_planning") or "") == "script_fallback"
    parse_max_tokens = large_json_max_tokens()
    # 任务启动时快照进 AgentState 的 Skill 方案决定系统提示词；节点不回读
    # 全局配置文件。整段解析与分段解析共用同一个已解析 Prompt。
    system_prompt = _load_system_prompt(state.get("skill_config"))

    try:
        parsed = await _parse_script(
            project_id,
            user_input,
            rag_context,
            effective_style,
            parse_max_tokens=parse_max_tokens,
            prefer_fallback=prefer_fallback,
            system_prompt=system_prompt,
        )
    except LLMOutputTruncatedError as exc:
        # 单次解析被截断后已尝试分段（见 _parse_script），分段仍截断说明单场内容
        # 就超出模型输出上限：这是确定性失败，保留诊断信息原样上抛。
        raise RuntimeError(f"Mimo 剧本解析失败: {exc}") from exc
    except LLMOutputError as exc:
        # 模型输出是不可信输入：统一走强 schema 校验，字段级问题在这里被归一化
        # 或丢弃；顶层结构错误转成可读业务错误，不会以裸 ValidationError 冒泡。
        raise RuntimeError(f"Mimo 剧本解析结果无法使用: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Mimo 剧本解析失败: {exc}") from exc

    characters = _build_characters(parsed.characters, effective_style)
    if not characters:
        raise RuntimeError("Mimo 剧本解析结果缺少可用角色")
    script_scenes = _build_scenes(parsed.script_scenes, characters)
    if not script_scenes:
        raise RuntimeError("Mimo 剧本解析结果缺少可用场景")

    project_memory.save_characters(project_id, characters)
    project_memory.save_narrative_context(
        project_id,
        {
            "script_scenes": script_scenes,
            "genre": parsed.genre or "原创短剧",
            "style_suggestion": effective_style,
        },
    )

    return {
        "script_title": parsed.title or _guess_title(user_input),
        "genre": parsed.genre or "原创短剧",
        "style_suggestion": effective_style,
        "requested_style": state.get("requested_style") or effective_style,
        "effective_style": effective_style,
        "style_source": state.get("style_source") or "project_request",
        "characters": characters,
        "raw_script": json.dumps(script_scenes, ensure_ascii=False),
        "script_scenes": script_scenes,
        "logic_issues": parsed.logic_issues,
        "rag_context": rag_context,
        "current_step": "parse_script",
    }


async def _parse_script(
    project_id: str,
    user_input: str,
    rag_context: list[str],
    effective_style: str,
    *,
    parse_max_tokens: int,
    prefer_fallback: bool = False,
    system_prompt: str = "",
) -> ScriptParseOutput:
    """解析剧本：短剧本单次调用；长剧本按场次边界分段解析后合并。

    单次调用被截断时自动降级为分段解析（改变了长度与分段策略，因此不是
    同配置重试）；分段调用仍被截断时对该段二分细化，直到低于安全下限才
    判定为确定性失败。分段结果逐段过统一 schema 校验，合并后再整体校验。
    """

    system_prompt = system_prompt or _load_system_prompt()
    whole_limit = max(1, int(settings.LLM_SCRIPT_PARSE_WHOLE_INPUT_CHARS))
    segment_limit = max(1, int(settings.LLM_SCRIPT_PARSE_SEGMENT_CHARS))
    min_chars = max(1, int(settings.LLM_SCRIPT_PARSE_SEGMENT_MIN_CHARS))
    force_segments = False
    if len(user_input) <= whole_limit:
        await _report_progress(project_id, "正在解析剧本", 8)
        try:
            result = await llm_service.call_json(
                system_prompt,
                _build_task_prompt(user_input, rag_context, effective_style),
                temperature=0.25,
                max_tokens=parse_max_tokens,
                prefer_fallback=prefer_fallback,
            )
            return parse_script_output(result, fallback_style=effective_style)
        except LLMOutputTruncatedError as exc:
            logger.warning(
                "剧本解析单次调用被截断，自动切换为分段解析: output_tokens=%s max_tokens=%s input_chars=%d",
                exc.output_tokens,
                exc.max_tokens,
                len(user_input),
            )
            # 截断后分段必须比原输入小，否则等于同配置重试。
            force_segments = True

    if force_segments:
        segment_limit = max(min_chars, min(segment_limit, len(user_input) // 2))
    segments = segment_script(user_input, limit=segment_limit)
    known_characters: list[str] = []
    outputs: list[ScriptParseOutput] = []
    total = len(segments)
    for index, segment in enumerate(segments, start=1):
        await _report_progress(project_id, f"正在解析剧本（第 {index}/{total} 段）", 8 + int(10 * index / total))
        segment_output = await _parse_segment_with_split(
            project_id,
            segment,
            known_characters,
            effective_style,
            parse_max_tokens=parse_max_tokens,
            prefer_fallback=prefer_fallback,
            min_chars=min_chars,
            progress=(index, total),
            system_prompt=system_prompt,
        )
        outputs.append(segment_output)
        for character in segment_output.characters:
            if character.name and character.name not in known_characters:
                known_characters.append(character.name)

    if not outputs:
        raise RuntimeError("剧本分段解析没有产生任何结果")
    return merge_script_outputs(outputs, fallback_style=effective_style)


async def _parse_segment_with_split(
    project_id: str,
    segment: str,
    known_characters: list[str],
    effective_style: str,
    *,
    parse_max_tokens: int,
    prefer_fallback: bool,
    min_chars: int,
    progress: tuple[int, int],
    system_prompt: str = "",
) -> ScriptParseOutput:
    """解析单个分段；段内仍被截断时按场次边界继续二分，低于下限才失败。"""

    system_prompt = system_prompt or _load_system_prompt()
    pending = [segment]
    split_attempts = 0
    while pending:
        text = pending.pop(0)
        try:
            result = await llm_service.call_json(
                system_prompt,
                _build_segment_prompt(text, known_characters, effective_style, progress),
                temperature=0.25,
                max_tokens=parse_max_tokens,
                prefer_fallback=prefer_fallback,
            )
            return parse_script_output(result, fallback_style=effective_style)
        except LLMOutputTruncatedError as exc:
            split_attempts += 1
            if len(text) <= min_chars or split_attempts > 4:
                raise
            index, total = progress
            logger.warning(
                "分段解析仍被截断（第 %d/%d 段，%d 字符），继续细分: output_tokens=%s",
                index,
                total,
                len(text),
                exc.output_tokens,
            )
            await _report_progress(project_id, f"正在解析剧本（第 {index}/{total} 段内容过长，已自动细分）", 18)
            pending = segment_script(text, limit=max(min_chars, len(text) // 2)) + pending
    raise RuntimeError("分段解析被截断且无法继续细分")


def segment_script(text: str, *, limit: int) -> list[str]:
    """按场次/章节自然边界把剧本切成不超过 limit 字符的分段。

    规则：
    - 优先在场景标题行边界切分，保证不把一场戏劈成两段；
    - 单个场景块本身超过 limit 时，在其内部按空行/换行边界切分；
    - 原文顺序完全保留，所有分段拼接等于原文（不丢字、不重复）。
    """

    raw = str(text or "")
    if not raw.strip():
        return []
    if len(raw) <= limit:
        return [raw]

    lines = raw.splitlines(keepends=True)
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if _SCENE_BOUNDARY.match(line) and current:
            blocks.append("".join(current))
            current = []
        current.append(line)
    if current:
        blocks.append("".join(current))
    if not blocks:
        blocks = [raw]

    segments: list[str] = []
    buffer = ""
    for block in blocks:
        if len(block) > limit:
            if buffer:
                segments.append(buffer)
                buffer = ""
            # 场景块过长：在块内按换行边界二次切分，避免无脑按字符硬切。
            part = ""
            for piece in re.split(r"(?<=\n)", block):
                if part and len(part) + len(piece) > limit:
                    segments.append(part)
                    part = piece
                else:
                    part += piece
            if part:
                segments.append(part)
            continue
        if buffer and len(buffer) + len(block) > limit:
            segments.append(buffer)
            buffer = block
        else:
            buffer += block
    if buffer:
        segments.append(buffer)

    merged: list[str] = []
    for segment in segments:
        # 相邻分段都低于 limit/2 时合并，避免产生大量碎段（保持顺序不变）。
        if merged and len(merged[-1]) + len(segment) <= limit:
            merged[-1] += segment
        else:
            merged.append(segment)
    return merged


def merge_script_outputs(outputs: list[ScriptParseOutput], *, fallback_style: str = "anime") -> ScriptParseOutput:
    """合并分段解析结果：去重角色/场景、保持原有顺序、场次重新连续编号。

    合并结果重新通过统一 schema 校验（ScriptParseOutput）后才交给调用方落库，
    因此分段合并不会产生重复角色、重复场景或乱序对白，也不会丢场次。
    """

    if not outputs:
        raise LLMOutputError("没有可合并的分段解析结果")
    title = next((item.title for item in outputs if item.title), "")
    genre = next((item.genre for item in outputs if item.genre), "")

    characters: list[CharacterOutput] = []
    character_index: dict[str, int] = {}
    for output in outputs:
        for character in output.characters:
            name = character.name.strip()
            if not name:
                continue
            if name in character_index:
                # 后续分段只用来补全首个角色卡片缺失的字段，不产生重复角色。
                existing = characters[character_index[name]]
                for field in ("personality", "visual_prompt", "negative_prompt", "voice_type", "default_outfit"):
                    if not getattr(existing, field) and getattr(character, field):
                        setattr(existing, field, getattr(character, field))
                if not existing.appearance:
                    existing.appearance = dict(character.appearance)
                if not existing.key_features:
                    existing.key_features = list(character.key_features)
                continue
            character_index[name] = len(characters)
            characters.append(character)

    def scene_signature(item: SceneOutput) -> tuple:
        first_line = item.dialogue[0].line if item.dialogue else ""
        return (item.location.strip(), item.actions.strip()[:120], first_line.strip()[:80])

    scenes: list[SceneOutput] = []
    seen_scenes: set[tuple] = set()
    for output in outputs:
        for scene in output.script_scenes:
            signature = scene_signature(scene)
            if signature in seen_scenes:
                continue
            seen_scenes.add(signature)
            scenes.append(scene)
    for number, scene in enumerate(scenes, start=1):
        scene.scene_number = number

    logic_issues: list[str] = []
    for output in outputs:
        for issue in output.logic_issues:
            text = str(issue).strip()
            if text and text not in logic_issues:
                logic_issues.append(text)

    # 组装为纯 dict 后统一走 ScriptParseOutput 校验：字段级校验器只接受 dict，
    # 直接传模型实例会被当作非法条目丢弃。
    merged_payload = {
        "title": title,
        "genre": genre,
        "style_suggestion": next((item.style_suggestion for item in outputs if item.style_suggestion), fallback_style),
        "characters": [character.model_dump() for character in characters[: settings.LLM_MAX_CHARACTERS]],
        "script_scenes": [scene.model_dump() for scene in scenes],
        "logic_issues": logic_issues[:20],
    }
    # 统一 schema 校验：分段合并后的结果与单次解析走同一套约束。
    return ScriptParseOutput.model_validate(merged_payload)


async def _report_progress(project_id: str, message: str, progress: int | None = None) -> None:
    """把解析进度写入任务中心（auto/manual 两个键都尝试，仅当前持有者生效）。"""

    try:
        from api.routes.script import _progress

        step = "parse_script"
        if progress is None:
            await _progress(project_id, step, 8, message)
        else:
            await _progress(project_id, step, max(0, min(100, int(progress))), message)
    except Exception:  # noqa: BLE001 - 进度上报失败不影响解析本身
        logger.debug("剧本解析进度上报失败: project=%s message=%s", project_id, message)


def _build_characters(items: list[CharacterOutput], effective_style: str = "anime") -> list[dict]:
    """把已校验的角色输出补全为角色卡片（音色、情绪变体、固定 seed）。"""

    characters: list[dict] = []
    for index, item in enumerate(items[: settings.LLM_MAX_CHARACTERS]):
        name = item.name.strip()
        if not name:
            continue
        characters.append(
            {
                "name": name,
                "appearance": dict(item.appearance),
                "personality": item.personality or "性格鲜明，行动目标清晰",
                "visual_prompt": item.visual_prompt or _default_visual_prompt(name, effective_style),
                "negative_prompt": item.negative_prompt or "low quality, blurry, watermark",
                "voice_id": normalize_mimo_voice(item.voice_type or "少女"),
                "key_features": item.key_features or _split_features(item.appearance.get("features", "")),
                "emotion_variants": dict(DEFAULT_EMOTION_VARIANTS),
                "seed": item.seed if item.seed is not None else 42 + index,
            }
        )
    return characters


def _split_features(raw: str) -> list[str]:
    """模型常把标志特征写成顿号/逗号分隔的字符串，这里拆成列表。"""

    return [part.strip() for part in re.split(r"[,，、]", raw or "") if part.strip()]


def _build_scenes(items: list[SceneOutput], characters: list[dict]) -> list[dict]:
    """把已校验的场景输出补全为剧本场景（对白、情绪、机位建议）。"""

    default_character = characters[0]["name"] if characters else "主角"
    scenes: list[dict] = []
    for index, item in enumerate(items[: settings.LLM_MAX_SCENES]):
        dialogue = [line.model_dump() for line in item.dialogue if line.line.strip()]
        scenes.append(
            {
                "scene_number": item.scene_number or index + 1,
                "location": item.location or "室内创作空间",
                "characters_in_scene": item.characters_in_scene or [default_character],
                "actions": item.actions or item.description or "角色推进剧情",
                "dialogue": dialogue,
                "emotion": item.emotion,
                "camera_suggestion": item.camera_suggestion,
            }
        )
    return scenes


def _guess_title(text: str) -> str:
    first = re.sub(r"\s+", "", text or "")[:14]
    return first or "未命名项目"


def _load_system_prompt(skill_config: dict | None = None) -> str:
    """解析剧本解析的系统提示词。

    Skill 方案自定义了 script_agent.system_prompt 时使用用户提示词并追加
    固定 JSON 输出契约；为空时返回与历史硬编码逐字一致的默认提示词。
    """
    return resolve_json_system_prompt(
        skill_config,
        "script_agent",
        fallback=SCRIPT_PARSE_SYSTEM_PROMPT,
        contract=SCRIPT_PARSE_JSON_CONTRACT,
    )


def _default_visual_prompt(name: str, effective_style: str) -> str:
    if effective_style == "realistic":
        return f"{name}, expressive live-action human portrait, natural skin texture, realistic wardrobe and anatomy"
    return f"{name}, expressive {effective_style} character reference, finished production design"


def _build_task_prompt(user_input: str, rag_context: list[str], effective_style: str = "anime") -> str:
    context = "\n\n参考内容：\n" + "\n---\n".join(rag_context) if rag_context else ""
    return f"""
项目实际生效画风（必须严格遵守，不要自行猜测或改写）：{effective_style}

请解析以下剧本或故事，输出 JSON：
{user_input}{context}

JSON 结构：
{{
  "title": "剧名",
  "genre": "类型",
  "style_suggestion": "{effective_style}",
  "characters": [
    {{
      "name": "角色名",
      "appearance": {{
        "hair": "发型发色",
        "eyes": "眼睛特征",
        "body": "身形",
        "features": "标志特征",
        "default_outfit": "默认服装"
      }},
      "personality": "性格",
      "visual_prompt": "English image prompt",
      "negative_prompt": "English negative prompt",
      "voice_type": "少女/少年/御姐/大叔/儿童/老人"
    }}
  ],
  "script_scenes": [
    {{
      "scene_number": 1,
      "location": "地点",
      "characters_in_scene": ["角色名"],
      "actions": "动作与剧情",
      "dialogue": [
        {{"character": "角色名", "line": "台词", "emotion": "neutral", "action": "说话动作"}}
      ],
      "emotion": "neutral/happy/shy/sad/angry/surprised",
      "camera_suggestion": "wide/medium/close-up/extreme_close"
    }}
  ],
  "logic_issues": []
}}
"""


def _build_segment_prompt(
    segment: str,
    known_characters: list[str],
    effective_style: str,
    progress: tuple[int, int],
) -> str:
    """长剧本分段解析的 prompt：只解析本段、沿用已知角色名、JSON 结构与整体解析一致。"""

    index, total = progress
    known = ("、".join(known_characters[:12])) if known_characters else "暂无"
    return f"""
项目实际生效画风（必须严格遵守，不要自行猜测或改写）：{effective_style}
这是一部长剧本的第 {index}/{total} 段。请只解析下面这一段文本，不要推测或补写其它段落的剧情。
已在前文出现过的角色（如本段再次出现请沿用完全相同的角色名，不要另起别名）：{known}

请输出 JSON：
{segment}

JSON 结构（与整体解析完全一致；title/genre 留空字符串，只输出本段内容）：
{{
  "title": "",
  "genre": "",
  "style_suggestion": "{effective_style}",
  "characters": [
    {{
      "name": "角色名",
      "appearance": {{
        "hair": "发型发色",
        "eyes": "眼睛特征",
        "body": "身形",
        "features": "标志特征",
        "default_outfit": "默认服装"
      }},
      "personality": "性格",
      "visual_prompt": "English image prompt",
      "negative_prompt": "English negative prompt",
      "voice_type": "少女/少年/御姐/大叔/儿童/老人"
    }}
  ],
  "script_scenes": [
    {{
      "scene_number": 1,
      "location": "地点",
      "characters_in_scene": ["角色名"],
      "actions": "动作与剧情",
      "dialogue": [
        {{"character": "角色名", "line": "台词", "emotion": "neutral", "action": "说话动作"}}
      ],
      "emotion": "neutral/happy/shy/sad/angry/surprised",
      "camera_suggestion": "wide/medium/close-up/extreme_close"
    }}
  ],
  "logic_issues": []
}}
"""
