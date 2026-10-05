from __future__ import annotations

from config import settings
from services.atomic_json import atomic_write_json, path_lock, read_json_file

STYLE_TEMPLATES: dict[str, dict[str, str]] = {
    "anime": {
        "label": "日系写实漫",
        "prompt_prefix": "Japanese realistic anime drama, refined cel shading, clean line art, natural skin tones, soft cinematic light",
        "video_prompt": "Japanese realistic anime short drama look, stable cel-shaded character identity, restrained cinematic camera movement",
        "character_reference_prompt": "Japanese realistic anime character design sheet, refined cel shading, full-body three-view reference",
        "scene_baseline_prompt": "Japanese realistic anime background key art, soft cinematic light, clean readable set dressing",
        "negative_prompt": "over-saturated neon, rough sketch, heavy grain, text artifacts, watermark",
    },
    "chinese": {
        "label": "国漫厚涂",
        "prompt_prefix": "Chinese animation painterly comic drama, elegant color blocks, soft brush texture, polished production keyframe",
        "video_prompt": "Chinese animation painterly short drama look, elegant palette, stable face and costume design",
        "character_reference_prompt": "Chinese animation painterly character design sheet, elegant facial features, full-body three-view reference",
        "scene_baseline_prompt": "Chinese animation painterly background key art, elegant light, clear prop placement",
        "negative_prompt": "flat plastic render, western superhero anatomy, messy brushwork, text artifacts, watermark",
    },
    "chibi": {
        "label": "简约条漫",
        "prompt_prefix": "clean vertical webcomic style, simplified cute proportions, crisp outlines, bright but controlled pastel palette",
        "video_prompt": "clean vertical webcomic short, cute simplified acting, stable pastel palette, gentle readable motion",
        "character_reference_prompt": "cute simplified webcomic character design sheet, consistent chibi proportions, full-body three-view reference",
        "scene_baseline_prompt": "clean webcomic background key art, simplified props, readable vertical composition",
        "negative_prompt": "realistic pores, horror lighting, cluttered detail, text artifacts, watermark",
    },
    "realistic": {
        "label": "电影写实",
        "prompt_prefix": "live-action cinematic realism, natural human anatomy, natural skin texture, live-action cinematic lighting, realistic fabric, real lens perspective, physically based materials, believable human proportions",
        "video_prompt": "live-action cinematic short drama, natural human anatomy, natural skin texture, live-action cinematic lighting, realistic fabric, real lens perspective, physically based materials, stable human identity and motion",
        "character_reference_prompt": "live-action cinematic human character reference sheet, natural human anatomy, natural skin texture, live-action cinematic lighting, realistic fabric, real lens perspective, physically based materials, full-body three-view reference",
        "scene_baseline_prompt": "live-action cinematic production background, live-action cinematic lighting, real lens perspective, physically based materials, realistic materials, natural spatial depth",
        "negative_prompt": "cartoon, anime, comic, chibi, cel shading, illustration, 3D render, plastic skin, deformed anatomy, text artifacts, watermark",
    },
    "watercolor": {
        "label": "水彩绘本",
        "prompt_prefix": "delicate watercolor storybook comic, translucent color wash, gentle paper texture, warm emotional atmosphere",
        "video_prompt": "watercolor storybook short drama look, soft color wash, gentle motion, stable illustrated character identity",
        "character_reference_prompt": "watercolor storybook character design sheet, soft paper texture, full-body three-view reference",
        "scene_baseline_prompt": "watercolor storybook background key art, airy atmosphere, clear prop silhouette",
        "negative_prompt": "muddy colors, heavy ink, photorealistic render, text artifacts, watermark",
    },
    "ink": {
        "label": "新国风水墨",
        "prompt_prefix": "modern Chinese ink-wash comic drama, restrained ink texture, mineral color accents, clean contemporary composition",
        "video_prompt": "modern Chinese ink-wash short drama, restrained brush texture, stable silhouettes, elegant camera rhythm",
        "character_reference_prompt": "modern Chinese ink-wash character design sheet, clean silhouette, full-body three-view reference",
        "scene_baseline_prompt": "modern Chinese ink-wash background key art, clear prop placement, elegant atmosphere",
        "negative_prompt": "old parchment cliché, chaotic ink splashes, unreadable scene, text artifacts, watermark",
    },
    "noir": {
        "label": "悬疑电影感",
        "prompt_prefix": "suspense cinematic comic drama, controlled contrast, muted color palette, precise motivated lighting",
        "video_prompt": "suspense cinematic short drama, controlled contrast, motivated light, stable spatial continuity",
        "character_reference_prompt": "suspense cinematic character design sheet, muted palette, full-body three-view reference",
        "scene_baseline_prompt": "suspense cinematic background key art, controlled shadows, clear perspective and prop continuity",
        "negative_prompt": "crushed blacks, horror gore, random neon, text artifacts, watermark",
    },
    "clay": {
        "label": "定格黏土",
        "prompt_prefix": "handcrafted clay stop-motion comic look, tactile material texture, miniature set lighting, warm studio palette",
        "video_prompt": "handcrafted clay stop-motion short drama, tactile motion, miniature set consistency, stable character shapes",
        "character_reference_prompt": "clay stop-motion character design sheet, tactile material, full-body three-view reference",
        "scene_baseline_prompt": "miniature clay set background key art, tactile props, clear prop positions",
        "negative_prompt": "plastic toy shine, melted faces, inconsistent scale, text artifacts, watermark",
    },
}


def style_template(style: str | None) -> dict[str, str]:
    key = (style or "anime").strip()
    return _all_templates().get(key, STYLE_TEMPLATES["anime"])


def style_prompt_params(style: str | None) -> dict[str, str]:
    template = style_template(style)
    return {
        "prompt_prefix": template["prompt_prefix"],
        "video_prompt": template["video_prompt"],
        "character_reference_prompt": template["character_reference_prompt"],
        "scene_baseline_prompt": template["scene_baseline_prompt"],
        "negative_prompt": template["negative_prompt"],
        "style_label": template["label"],
    }


def style_options() -> list[dict[str, str]]:
    return [
        {
            "value": key,
            "label": value["label"],
            "keywords": value.get("keywords", ""),
            "custom": key.startswith("custom_"),
        }
        for key, value in _all_templates().items()
    ]


def create_custom_style_template(
    key: str,
    label: str,
    keywords: str,
    negative_prompt: str = "",
    created_at: str = "",
) -> dict[str, str]:
    template = {
        "label": label,
        "keywords": keywords,
        "prompt_prefix": keywords,
        "video_prompt": f"{keywords}, stable short drama video style, consistent palette and line language",
        "character_reference_prompt": f"{keywords}, character design sheet, full-body three-view reference",
        "scene_baseline_prompt": f"{keywords}, background key art, clear scene baseline reference",
        "negative_prompt": negative_prompt or "low quality, watermark, text artifacts, inconsistent style",
        "created_at": created_at,
    }
    path = _custom_template_path()
    with path_lock(path):
        custom = _custom_templates()
        custom[key] = template
        atomic_write_json(path, custom)
    return {"value": key, "label": label, "keywords": keywords, "custom": True}


def _all_templates() -> dict[str, dict[str, str]]:
    return {**STYLE_TEMPLATES, **_custom_templates()}


def _custom_templates() -> dict[str, dict[str, str]]:
    data = read_json_file(_custom_template_path(), default=None)
    return data if isinstance(data, dict) else {}


def _custom_template_path():
    return settings.DATA_DIR / "custom_style_templates.json"
