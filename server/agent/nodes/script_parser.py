import json
import re

from agent.output_schemas import CharacterOutput, LLMOutputError, SceneOutput, parse_script_output
from agent.state import AgentState
from config import settings
from memory.project_memory import ProjectMemory
from rag.rag_service import RAGService
from services.llm_service import LLMService
from services.tts_service import normalize_mimo_voice

llm_service = LLMService()
rag_service = RAGService()
project_memory = ProjectMemory()

DEFAULT_EMOTION_VARIANTS = {
    "neutral": "calm expression",
    "happy": "gentle smile",
    "shy": "slight blush",
    "sad": "downcast eyes",
    "angry": "determined eyes",
    "surprised": "wide eyes",
}


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

    try:
        result = await llm_service.call_json(
            _load_system_prompt(),
            _build_task_prompt(user_input, rag_context, state.get("effective_style") or state.get("style", "anime")),
            temperature=0.25,
        )
    except Exception as exc:
        raise RuntimeError(f"Mimo 剧本解析失败: {exc}") from exc

    # 模型输出是不可信输入：统一走强 schema 校验，字段级问题在这里被归一化或丢弃，
    # 不会以 KeyError/TypeError/ValueError 的形式冒泡到流水线。
    try:
        parsed = parse_script_output(result, fallback_style=state.get("style", "anime"))
    except LLMOutputError as exc:
        raise RuntimeError(f"Mimo 剧本解析结果无法使用: {exc}") from exc

    effective_style = state.get("effective_style") or state.get("style") or "anime"
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


def _load_system_prompt() -> str:
    return (
        "你是资深漫剧编导。请把用户输入解析成角色、场景、对白和情绪，"
        "输出严格 JSON，不要输出 Markdown。"
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
