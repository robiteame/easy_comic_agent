"""写实风格 Prompt 语义验收测试。

- realistic 正向词必须是真人电影写实语义，且不得包含 comic/anime/cartoon/
  chibi/cel shading/illustration 等冲突词；
- realistic 负向词必须覆盖卡通/动漫/3D 渲染/塑料皮肤/畸形解剖/文字伪影/水印；
- 所有风格的模板输出不得再包含统一追加的 ``vertical cinematic comic frame``；
- 图像服务用 realistic 风格参数构造的 prompt 同样满足上述约束。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services.image_service import ImageService  # noqa: E402
from services.style_templates import STYLE_TEMPLATES, style_prompt_params, style_template  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402

POSITIVE_CONFLICT_WORDS = ("comic", "anime", "cartoon", "chibi", "cel shading", "illustration")
REQUIRED_POSITIVE_TERMS = (
    "natural human anatomy",
    "natural skin texture",
    "realistic fabric",
    "live-action cinematic lighting",
    "real lens perspective",
    "physically based materials",
)
REQUIRED_NEGATIVE_TERMS = (
    "cartoon",
    "anime",
    "comic",
    "chibi",
    "cel shading",
    "3d render",
    "plastic skin",
    "deformed anatomy",
    "text artifacts",
    "watermark",
)


class RealisticTemplateTests(unittest.TestCase):
    def test_positive_terms_present(self) -> None:
        template = style_template("realistic")
        for field in ("prompt_prefix", "video_prompt", "character_reference_prompt", "scene_baseline_prompt"):
            text = template[field].lower()
            for term in REQUIRED_POSITIVE_TERMS:
                if field == "scene_baseline_prompt" and term in {
                    "natural human anatomy",
                    "natural skin texture",
                    "realistic fabric",
                }:
                    continue  # 场景基准图不涉及人体/服装词
                self.assertIn(term, text, f"{field} 缺少写实正向词 {term}")

    def test_positive_terms_have_no_conflict_words(self) -> None:
        template = style_template("realistic")
        for field in ("prompt_prefix", "video_prompt", "character_reference_prompt", "scene_baseline_prompt"):
            text = template[field].lower()
            for word in POSITIVE_CONFLICT_WORDS:
                self.assertNotIn(word, text, f"{field} 正向词包含冲突词 {word}")

    def test_negative_terms_complete(self) -> None:
        negative = style_template("realistic")["negative_prompt"].lower()
        for term in REQUIRED_NEGATIVE_TERMS:
            self.assertIn(term, negative)

    def test_no_template_appends_vertical_cinematic_comic_frame(self) -> None:
        for key, template in STYLE_TEMPLATES.items():
            joined = ", ".join(str(value) for value in template.values()).lower()
            self.assertNotIn("vertical cinematic comic frame", joined, f"风格 {key} 仍附加 comic frame 文案")

    def test_anime_template_still_uses_anime_semantics(self) -> None:
        # anime 模板自身仍是动漫语义（写实约束只针对 realistic）。
        self.assertIn("anime", style_template("anime")["prompt_prefix"].lower())


class RealisticServicePromptTests(unittest.TestCase):
    def test_shot_prompt_with_realistic_style_has_no_comic_words(self) -> None:
        service = ImageService()
        shot = {
            "shot_id": "p_shot_0001",
            "scene_description": "雨夜天台，霓虹灯反光",
            "character_action": "她攥紧伞柄转身",
            "visual_notes": "",
            "storyboard_prompt": "approved storyboard",
            "shot_type": "medium",
            "camera_angle": "正面",
            "camera_movement": "缓慢推进",
            "emotion": "sad",
        }
        characters = [
            {
                "id": "c1",
                "name": "林晚",
                "visual_prompt": "Lin Wan, 26-year-old East Asian woman, oval face, black long hair, beige trench coat",
                "key_features": ["黑长直", "米色风衣"],
                "appearance": {"hair": "black long hair", "outfit": "beige trench coat"},
                "reference_images": ["/tmp/three_view.png"],
                "wardrobe_lock": "LOCKED wardrobe: beige trench coat",
                "emotion_variants": {"sad": "downcast eyes"},
            }
        ]
        prompt, negative = service._build_prompt(shot, characters, style_prompt_params("realistic"))
        lowered = prompt.lower()
        for word in POSITIVE_CONFLICT_WORDS:
            self.assertNotIn(word, lowered, f"写实正向 Prompt 包含冲突词 {word}")
        for term in ("natural human anatomy", "live-action cinematic"):
            self.assertIn(term, lowered)
        self.assertIn("lin wan", lowered)
        self.assertIn("beige trench coat", lowered)
        negative_lower = negative.lower()
        for term in ("cartoon", "anime", "comic", "watermark"):
            self.assertIn(term, negative_lower)

    def test_script_parser_default_visual_prompt_is_style_aware(self) -> None:
        from agent.nodes.script_parser import _default_visual_prompt

        realistic = _default_visual_prompt("林晚", "realistic")
        self.assertIn("live-action", realistic.lower())
        for word in POSITIVE_CONFLICT_WORDS:
            self.assertNotIn(word, realistic.lower())
        anime = _default_visual_prompt("林晚", "anime")
        # anime 风格默认描述本身允许 anime 语义，但不再是写死的 comic character design。
        self.assertNotIn("comic character design", anime.lower())


if __name__ == "__main__":
    unittest.main()
