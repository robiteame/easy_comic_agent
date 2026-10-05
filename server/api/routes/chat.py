from typing import Annotated

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.schemas import GenerationPrompt, Identifier
from services.llm_service import LLMService

router = APIRouter(prefix="/api/chat", tags=["chat"])
llm_service = LLMService()


class ChatRequest(BaseModel):
    project_id: Identifier
    message: GenerationPrompt
    current_shots: Annotated[list[dict], Field(max_length=100)] = []


@router.post("")
async def chat_with_agent(data: ChatRequest):
    """自然语言交互 - 修改镜头"""

    system_prompt = """你是用户的漫剧创作助手，通过自然语言交互帮助用户调整漫剧。

你拥有以下能力：
1. 修改镜头内容（角色动作、表情、场景、台词）
2. 调整镜头参数（类型、时长、角度）
3. 重新生成指定镜头
4. 批量调整多个镜头
5. 回答创作相关问题

请根据用户的指令，输出JSON格式的操作建议：
{
    "response": "回复用户的文字",
    "actions": [
        {
            "type": "update_shot",
            "shot_id": "shot_0001",
            "changes": {
                "emotion": "shy",
                "scene_description": "..."
            }
        }
    ],
    "need_confirm": true
}"""

    shots_summary = ""
    if data.current_shots:
        shots_summary = "当前镜头列表：\n"
        for s in data.current_shots[:10]:
            shots_summary += (
                f"- {s.get('id', '?')}: {s.get('shot_type', '?')} | {s.get('scene_description', '')[:50]}\n"
            )

    user_prompt = f"{shots_summary}\n用户指令：{data.message}"

    try:
        result = await llm_service.call_json(system_prompt, user_prompt)
        return result
    except Exception as e:
        return {
            "response": f"处理失败: {str(e)}",
            "actions": [],
            "need_confirm": False,
        }
