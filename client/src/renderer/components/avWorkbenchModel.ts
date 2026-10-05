/**
 * 字幕与音频工作台的纯逻辑层：时间线换算、吸附、本地告警与撤销栈。
 *
 * 不依赖 DOM 与 React，组件只做渲染与事件绑定；所有可测试的规则都收敛在
 * 这里（与后端 analyze 接口的规则保持同义：重叠 / 越界 / 静音 / 无伴奏轨）。
 */

import type { AudioTrackDto, AvShotInfo, AvWarning, SubtitleCueDto } from '../services/api'

/** 时间线缩放档位（像素/秒）。 */
export const TIMELINE_ZOOM_STEPS = [24, 40, 64, 100, 160] as const
export const DEFAULT_ZOOM_INDEX = 1
/** 播放头与片段边缘的吸附阈值（毫秒）。 */
export const SNAP_MS = 120

export const TRACK_KIND_LABELS: Record<string, string> = {
  dialogue: '对白',
  music: '背景音乐',
  ambient: '环境音',
  sfx: '音效',
}

export const TRACK_KIND_ORDER: Array<'dialogue' | 'music' | 'ambient' | 'sfx'> = ['dialogue', 'music', 'ambient', 'sfx']

export function msToPx(ms: number, pxPerSecond: number): number {
  return (Math.max(0, ms) / 1000) * pxPerSecond
}

export function pxToMs(px: number, pxPerSecond: number): number {
  if (pxPerSecond <= 0) return 0
  return Math.max(0, Math.round((px / pxPerSecond) * 1000))
}

/** 把毫秒吸附到最近的网格 / 片段边缘 / 镜头边界。 */
export function snapMs(ms: number, candidates: Array<number>, thresholdMs: number = SNAP_MS): number {
  let best = ms
  let bestDistance = Number.POSITIVE_INFINITY
  for (const candidate of candidates) {
    const distance = Math.abs(candidate - ms)
    if (distance < bestDistance && distance <= thresholdMs) {
      best = candidate
      bestDistance = distance
    }
  }
  return best
}

export function clampMs(ms: number, lowMs: number, highMs: number): number {
  return Math.min(Math.max(ms, lowMs), Math.max(lowMs, highMs))
}

export function formatTimelineMs(ms: number): string {
  const total = Math.max(0, Math.floor(ms))
  const minutes = Math.floor(total / 60_000)
  const seconds = Math.floor((total % 60_000) / 1000)
  const decis = Math.floor((total % 1000) / 100)
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${decis}`
}

/** 镜头区间（含镜头间隙为 0 的假设下按顺序累积）。 */
export function computeShotSpans(shotInfos: AvShotInfo[]): Map<string, { start_ms: number; end_ms: number }> {
  const spans = new Map<string, { start_ms: number; end_ms: number }>()
  for (const info of shotInfos) spans.set(info.id, { start_ms: info.start_ms, end_ms: info.end_ms })
  return spans
}

export interface TimelineTrackClip {
  track: AudioTrackDto
  /** 时间线上的起点（含 delay；对白轨来自镜头区间）。 */
  start_ms: number
  /** 播放长度（循环轨铺满全片）。 */
  duration_ms: number
  /** 与全片的相对关系，用于着色。 */
  outOfRange: boolean
}

/** 把轨道换算成时间线片段：与后端混音规划器同一套区间规则。 */
export function buildTrackClips(tracks: AudioTrackDto[], totalDurationMs: number): TimelineTrackClip[] {
  return tracks.map((track) => {
    const span = track.kind === 'dialogue' ? track.shot_span : undefined
    const start = (span ? span.start_ms : track.start_ms) + Math.max(0, track.delay_ms || 0)
    const trimmed = (track.source_duration_ms || 0) - (track.trim_start_ms || 0) - (track.trim_end_ms || 0)
    const duration = track.loop ? Math.max(0, totalDurationMs - start) : Math.max(0, trimmed)
    return {
      track,
      start_ms: start,
      duration_ms: duration,
      outOfRange: totalDurationMs > 0 && start >= totalDurationMs,
    }
  })
}

export interface LocalWarningInput {
  tracks: AudioTrackDto[]
  clips: TimelineTrackClip[]
  shotInfos: AvShotInfo[]
  cues: SubtitleCueDto[]
  totalDurationMs: number
}

/** 本地即时告警（保存前提示；权威判定仍以后端 analyze 为准）。 */
export function computeLocalWarnings(input: LocalWarningInput): AvWarning[] {
  const warnings: AvWarning[] = []
  const { tracks, clips, shotInfos, cues, totalDurationMs } = input
  const shotById = new Map(shotInfos.map((info) => [info.id, info]))

  for (const track of tracks) {
    const label = track.name || TRACK_KIND_LABELS[track.kind] || track.id
    if (track.muted) {
      warnings.push({
        level: 'warning',
        code: 'muted',
        message: `轨道「${label}」已静音，不参与混音`,
        track_id: track.id,
      })
    }
    if (track.kind === 'dialogue') {
      const shot = track.shot_id ? shotById.get(track.shot_id) : undefined
      if (!shot) {
        warnings.push({
          level: 'error',
          code: 'dialogue_unbound',
          message: `对白轨「${label}」未绑定有效镜头`,
          track_id: track.id,
        })
      } else if (!shot.has_tts && !shot.native_audio) {
        warnings.push({
          level: 'warning',
          code: 'dialogue_no_tts',
          message: `对白轨「${label}」绑定的镜头还没有配音（对白静音）`,
          track_id: track.id,
        })
      }
    } else if (!track.source_path) {
      warnings.push({
        level: 'error',
        code: 'source_missing',
        message: `轨道「${label}」缺少素材，会被剔除出混音`,
        track_id: track.id,
      })
    }
    if (totalDurationMs > 0 && (track.start_ms || 0) >= totalDurationMs && track.kind !== 'dialogue') {
      warnings.push({
        level: 'error',
        code: 'out_of_range',
        message: `轨道「${label}」起点超出全片时长`,
        track_id: track.id,
      })
    }
    if ((track.volume || 0) > 1.5 && !track.muted) {
      warnings.push({
        level: 'warning',
        code: 'hot_gain',
        message: `轨道「${label}」音量偏大（${(track.volume || 0).toFixed(2)}），注意削波`,
        track_id: track.id,
      })
    }
  }

  // 同类型轨道的时间重叠。
  const byKind = new Map<string, TimelineTrackClip[]>()
  for (const clip of clips) {
    if (clip.track.muted) continue
    const list = byKind.get(clip.track.kind) || []
    list.push(clip)
    byKind.set(clip.track.kind, list)
  }
  for (const [kind, list] of byKind) {
    const ordered = [...list].sort((a, b) => a.start_ms - b.start_ms)
    for (let i = 1; i < ordered.length; i += 1) {
      const previous = ordered[i - 1]
      const current = ordered[i]
      if (current.start_ms < previous.start_ms + previous.duration_ms) {
        warnings.push({
          level: 'warning',
          code: 'overlap',
          message: `${TRACK_KIND_LABELS[kind] || kind}轨「${current.track.name || current.track.id}」与上一条时间重叠`,
          track_id: current.track.id,
        })
      }
    }
  }

  // 字幕越界与相邻重叠。
  for (let i = 0; i < cues.length; i += 1) {
    const cue = cues[i]
    if (totalDurationMs > 0 && cue.end_ms > totalDurationMs + 500) {
      warnings.push({ level: 'warning', code: 'subtitle_beyond', message: `第 ${i + 1} 条字幕超出全片时长` })
      break
    }
  }
  for (let i = 1; i < cues.length; i += 1) {
    if (cues[i].start_ms < cues[i - 1].end_ms) {
      warnings.push({ level: 'warning', code: 'subtitle_overlap', message: `第 ${i}、${i + 1} 条字幕时间重叠` })
      break
    }
  }

  if (totalDurationMs > 0 && !tracks.some((t) => t.kind !== 'dialogue' && !t.muted)) {
    warnings.push({ level: 'info', code: 'no_bed', message: '尚未配置音乐 / 环境音 / 音效轨，成片只有对白与环境底噪' })
  }
  return warnings
}

/** 播放头命中的第一条字幕（预览叠加显示用；草稿与服务端条目都适用）。 */
export function cueAtPlayhead(
  cues: Array<Pick<SubtitleCueDto, 'start_ms' | 'end_ms' | 'text'>>,
  playheadMs: number,
): Pick<SubtitleCueDto, 'start_ms' | 'end_ms' | 'text'> | null {
  for (const cue of cues) {
    if (playheadMs >= cue.start_ms && playheadMs < cue.end_ms) return cue
  }
  return null
}

/** 播放头附近的吸附候选：镜头边界 + 字幕边缘 + 片段边缘 + 秒网格。 */
export function snapCandidates(
  shotInfos: AvShotInfo[],
  clips: TimelineTrackClip[],
  cues: SubtitleCueDto[],
  playheadMs: number,
): number[] {
  const candidates: number[] = []
  for (const info of shotInfos) candidates.push(info.start_ms, info.end_ms)
  for (const clip of clips) candidates.push(clip.start_ms, clip.start_ms + clip.duration_ms)
  for (const cue of cues) candidates.push(cue.start_ms, cue.end_ms)
  const second = Math.round(playheadMs / 1000) * 1000
  candidates.push(second, second - 1000, second + 1000)
  return candidates
}

export interface UndoSnapshot<T> {
  label: string
  state: T
}

const MAX_UNDO_DEPTH = 40

/** 固定深度撤销栈：push 返回新栈（不可变，方便配合 React state）。 */
export function pushUndo<T>(stack: UndoSnapshot<T>[], label: string, state: T): UndoSnapshot<T>[] {
  const next = [...stack, { label, state }]
  return next.length > MAX_UNDO_DEPTH ? next.slice(next.length - MAX_UNDO_DEPTH) : next
}

export function undoOnce<T, R>(
  stack: UndoSnapshot<T>[],
  apply: (state: T) => R,
): { stack: UndoSnapshot<T>[]; result: R | null } {
  // 栈顶是当前状态：撤销 = 丢弃栈顶并回到新的栈顶；只剩初始快照时不可撤销。
  if (stack.length <= 1) return { stack, result: null }
  const next = stack.slice(0, -1)
  return { stack: next, result: apply(next[next.length - 1].state) }
}

export function canUndo<T>(stack: UndoSnapshot<T>[]): boolean {
  return stack.length > 1
}

/** 字幕草稿的本地预校验（与服务端 subtitle_service 规则同义，提示但不拦截输入）。 */
export function validateCueDraft(cue: { text: string; start_ms: number; end_ms: number }): string | null {
  if (!cue.text.trim()) return '字幕文本不能为空'
  if (cue.text.length > 500) return '字幕文本过长（上限 500 字符）'
  // 与服务端 _FORBIDDEN_CHARS 同义：C0 控制字符（保留 \t\n）、DEL、行/段分隔符。
  // biome-ignore lint/suspicious/noControlCharactersInRegex: 该正则的职责就是匹配这些控制字符以拒绝它们
  if (/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u2028\u2029]/.test(cue.text)) return '字幕文本包含非法控制字符'
  if (cue.end_ms <= cue.start_ms) return '结束时间必须晚于开始时间'
  if (cue.end_ms - cue.start_ms < 50) return '字幕时长不足 50 毫秒'
  return null
}

/** 轨道分道：按 kind 分组并保持 order_index 顺序。 */
export function groupTracksByKind(
  tracks: AudioTrackDto[],
): Array<{ kind: 'dialogue' | 'music' | 'ambient' | 'sfx'; items: AudioTrackDto[] }> {
  const groups = TRACK_KIND_ORDER.map((kind) => ({ kind, items: [] as AudioTrackDto[] }))
  const indexByKind = new Map(TRACK_KIND_ORDER.map((kind, index) => [kind, index]))
  for (const track of [...tracks].sort((a, b) => (a.order_index || 0) - (b.order_index || 0))) {
    const group = indexByKind.get(track.kind)
    if (group !== undefined) groups[group].items.push(track)
  }
  return groups.filter((group) => group.items.length > 0)
}

/** 数字输入的安全解析（NaN / 负数回退默认值）。 */
export function parseMsInput(value: string, fallback: number): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) return fallback
  return Math.min(Math.round(parsed), 86_400_000)
}
