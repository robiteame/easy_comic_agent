from agent.state import AgentState
from services.dialogue_audio import generate_dialogue_track
from services.shot_dialogue import parse_shot_dialogue


async def run(state: AgentState) -> dict:
    """Generate voice audio for shots that contain dialogue.

    每句对白按自身 speaker 选择角色音色逐句合成后拼接；说话人不在角色列表
    时由 resolve_speaker_voice 记录可追踪警告并使用端点默认音色。
    """

    shots = state.get("shots", [])
    characters = state.get("characters", [])
    project_id = state["project_id"]

    updated_shots = []
    failures: list[str] = []
    for shot in shots:
        lines = [
            line
            for line in parse_shot_dialogue(shot.get("dialogue"), default_emotion=shot.get("emotion", "neutral"))
            if line.line.strip()
        ]
        if not lines:
            updated_shots.append(shot)
            continue

        try:
            audio_path, timed_lines = await generate_dialogue_track(
                lines,
                characters=characters,
                project_id=project_id,
                media_id=shot["shot_id"],
                default_emotion=shot.get("emotion", "neutral"),
            )
            shot["audio_path"] = audio_path
            shot["dialogue"] = [line.as_dict() for line in timed_lines]
        except Exception as e:
            shot["audio_path"] = ""
            shot["status"] = "needs_review"
            shot["visual_notes"] = f"配音失败: {str(e)}"
            failures.append(f"{shot.get('shot_id')}: {e}")

        updated_shots.append(shot)

    if failures:
        raise RuntimeError("配音生成失败: " + " | ".join(failures))

    return {
        "shots": updated_shots,
        "current_step": "generate_voice",
    }
