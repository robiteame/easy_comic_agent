"""字幕 / 音频配置的读取、汇总与失效。

工作台修改任何轨道或字幕条目时调用 bump_av_config_version；渲染任务在开始时
用 build_av_manifest 记录当时配置，发布前重新构建比对，不一致则丢弃成片。
collect_render_config 把 DB 行转成 FFmpegService 需要的 dict（含镜头区间、
TTS 素材路径与时长），渲染与预览共用，保证两条管线输入一致。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from sqlalchemy.orm import Session

from models import AudioTrack, Project, Shot, SubtitleCue, SubtitleTrack
from services.security import existing_file


@dataclass
class AvRenderConfig:
    """一次渲染 / 预览所需的全部字幕与音频数据（与 ORM 解耦）。"""

    audio_tracks: list[dict]
    subtitle_tracks: list[dict]
    # 镜头在时间线上的区间（毫秒），dialogue 轨与字幕生成都依赖它。
    shot_spans: dict[str, tuple[int, int]]
    total_duration_ms: int


def _shot_spans(db: Session, project_id: str) -> dict[str, tuple[int, int]]:
    shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.sequence).all()
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for shot in shots:
        duration_ms = int(round(max(0.0, float(shot.duration or 0.0)) * 1000))
        spans[shot.id] = (cursor, cursor + duration_ms)
        cursor += duration_ms
    return spans


def bump_av_config_version(db: Session, project: Project) -> int:
    """推进字幕/音频配置版本；成片已发布时同步标记为待重渲。"""

    project.av_config_version = int(project.av_config_version or 0) + 1
    if project.status == "completed":
        project.status = "assets_ready"
    db.commit()
    return project.av_config_version


def _serialize_audio_track(track: AudioTrack, shot_spans: dict[str, tuple[int, int]]) -> dict:
    start_ms = int(track.start_ms or 0)
    shot_span: tuple[int, int] | None = None
    if track.kind == "dialogue" and track.shot_id:
        shot_span = shot_spans.get(track.shot_id)
        if shot_span is not None:
            start_ms = shot_span[0]
    return {
        "id": track.id,
        "kind": track.kind,
        "name": track.name or "",
        "source_path": track.source_path or "",
        "source_duration_ms": int(track.source_duration_ms or 0),
        "shot_id": track.shot_id or "",
        "start_ms": start_ms,
        "volume": float(track.volume if track.volume is not None else 1.0),
        "pan": float(track.pan or 0.0),
        "fade_in_ms": int(track.fade_in_ms or 0),
        "fade_out_ms": int(track.fade_out_ms or 0),
        "delay_ms": int(track.delay_ms or 0),
        "trim_start_ms": int(track.trim_start_ms or 0),
        "trim_end_ms": int(track.trim_end_ms or 0),
        "loop": bool(track.loop),
        "muted": bool(track.muted),
        "duck_amount_db": float(track.duck_amount_db or 0.0),
        "duck_attack_ms": int(track.duck_attack_ms or 120),
        "duck_release_ms": int(track.duck_release_ms or 480),
        "order_index": int(track.order_index or 0),
        # dialogue 轨的素材在渲染时解析为 shot.audio_path（TTS 配音）。
        "shot_span": shot_span,
    }


def _serialize_subtitle_track(track: SubtitleTrack, cues: list[SubtitleCue]) -> dict:
    return {
        "id": track.id,
        "name": track.name or "",
        "language": track.language or "zh",
        "burn_in": bool(track.burn_in),
        "enabled": bool(track.enabled),
        "font_family": track.font_family or "sans-serif",
        "font_size": int(track.font_size or 54),
        "primary_color": track.primary_color or "#FFFFFF",
        "outline_color": track.outline_color or "#000000",
        "outline_width": int(track.outline_width or 0),
        "bold": bool(track.bold),
        "position": track.position or "bottom",
        "safe_margin": int(track.safe_margin or 0),
        "cues": [
            {
                "start_ms": int(cue.start_ms or 0),
                "end_ms": int(cue.end_ms or 0),
                "text": cue.text or "",
                "character_name": cue.character_name or "",
            }
            for cue in cues
        ],
    }


def collect_render_config(db: Session, project_id: str) -> AvRenderConfig:
    """读取工作台配置并解析 dialogue 轨素材（TTS 路径 + 有效时长）。"""

    shot_spans = _shot_spans(db, project_id)
    tracks = (
        db.query(AudioTrack)
        .filter(AudioTrack.project_id == project_id)
        .order_by(AudioTrack.order_index, AudioTrack.created_at)
        .all()
    )
    media_roots = _media_roots()
    audio_dicts: list[dict] = []
    for track in tracks:
        item = _serialize_audio_track(track, shot_spans)
        if track.kind == "dialogue":
            shot = db.query(Shot).filter(Shot.id == track.shot_id, Shot.project_id == project_id).first()
            item["resolved_source_path"] = shot.audio_path or "" if shot else ""
            span = item.get("shot_span")
            item["clip_limit_ms"] = (span[1] - span[0]) if span else 0
        else:
            item["resolved_source_path"] = track.source_path or ""
            item["clip_limit_ms"] = 0
        # 无效素材（文件缺失 / 越出允许根 / 时长未知）在此处剔除并留 warning。
        resolved = existing_file(item["resolved_source_path"], minimum_size=1, allowed_roots=media_roots)
        if resolved is None:
            item["resolved_source_path"] = ""
        audio_dicts.append(item)

    subtitle_rows = (
        db.query(SubtitleTrack).filter(SubtitleTrack.project_id == project_id).order_by(SubtitleTrack.created_at).all()
    )
    track_ids = [row.id for row in subtitle_rows]
    cue_rows: list[SubtitleCue] = []
    if track_ids:
        cue_rows = (
            db.query(SubtitleCue)
            .filter(SubtitleCue.track_id.in_(track_ids))
            .order_by(SubtitleCue.track_id, SubtitleCue.order_index)
            .all()
        )
    cues_by_track: dict[str, list[SubtitleCue]] = {}
    for cue in cue_rows:
        cues_by_track.setdefault(cue.track_id, []).append(cue)
    subtitle_dicts = [_serialize_subtitle_track(row, cues_by_track.get(row.id, [])) for row in subtitle_rows]

    total_ms = max((span[1] for span in shot_spans.values()), default=0)
    return AvRenderConfig(
        audio_tracks=audio_dicts,
        subtitle_tracks=subtitle_dicts,
        shot_spans=shot_spans,
        total_duration_ms=total_ms,
    )


def _media_roots() -> tuple:
    from config import settings

    return (settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR)


def build_av_manifest(av_config: AvRenderConfig) -> dict:
    """配置摘要：渲染开始时记录、发布前重建比对。"""

    def _track_entry(item: dict) -> tuple:
        source_ref = ""
        if item.get("resolved_source_path"):
            source_ref = hashlib.sha1(str(item["resolved_source_path"]).encode("utf-8")).hexdigest()[:12]
        return (
            item["id"],
            item["kind"],
            source_ref,
            item["start_ms"],
            item["delay_ms"],
            round(float(item["volume"]), 4),
            round(float(item["pan"]), 4),
            item["fade_in_ms"],
            item["fade_out_ms"],
            item["trim_start_ms"],
            item["trim_end_ms"],
            bool(item["loop"]),
            bool(item["muted"]),
            round(float(item["duck_amount_db"]), 2),
        )

    def _subtitle_entry(item: dict) -> tuple:
        return (
            item["id"],
            item["burn_in"],
            item["enabled"],
            item["font_family"],
            item["font_size"],
            item["primary_color"],
            item["outline_color"],
            item["outline_width"],
            bool(item["bold"]),
            item["position"],
            item["safe_margin"],
            tuple((cue["start_ms"], cue["end_ms"], cue["text"], cue["character_name"]) for cue in item["cues"]),
        )

    return {
        "audio": [_track_entry(item) for item in av_config.audio_tracks],
        "subtitle": [_subtitle_entry(item) for item in av_config.subtitle_tracks],
    }


def manifest_payload(manifest: dict) -> str:
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=list)


def av_manifest_from_db(db: Session, project_id: str) -> dict:
    """便捷入口：读取配置并直接生成比对用 manifest。"""

    return build_av_manifest(collect_render_config(db, project_id))


__all__ = [
    "AvRenderConfig",
    "av_manifest_from_db",
    "build_av_manifest",
    "bump_av_config_version",
    "collect_render_config",
    "manifest_payload",
]
