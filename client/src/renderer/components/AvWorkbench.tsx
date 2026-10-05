import type React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Button from 'antd/es/button'
import Input from 'antd/es/input'
import message from 'antd/es/message'
import Select from 'antd/es/select'
import Switch from 'antd/es/switch'
import { AudioOutlined, DeleteOutlined, DownloadOutlined, UndoOutlined, UploadOutlined } from '@ant-design/icons'

import {
  audioTrackApi,
  subtitleApi,
  toOutputUrl,
  type AudioTrackDto,
  type AudioTrackKind,
  type AudioTrackPayload,
  type AvShotInfo,
  type AvWarning,
  type MixPreviewStatus,
  type SubtitleTrackDto,
} from '../services/api'
import { useProjectStore } from '../stores/projectStore'
import {
  TRACK_KIND_LABELS,
  type UndoSnapshot,
  buildTrackClips,
  canUndo,
  clampMs,
  computeLocalWarnings,
  cueAtPlayhead,
  formatTimelineMs,
  groupTracksByKind,
  msToPx,
  pushUndo,
  snapCandidates,
  snapMs,
  undoOnce,
  validateCueDraft,
} from './avWorkbenchModel'

const _LANE_HEIGHT = 40
const _LABEL_COLUMN_WIDTH = 92
const TIMELINE_PADDING_MS = 1500

type CueDraft = {
  key: string
  start_ms: number
  end_ms: number
  text: string
  character_name: string
}

type UndoAction =
  | { kind: 'cues'; label: string; cues: CueDraft[] }
  | { kind: 'track'; label: string; trackId: string; params: Record<string, unknown> }

const AUDIO_ACCEPT = '.mp3,.wav,.m4a,.aac,.ogg,.flac,.opus,.wma,.aiff'
const SUBTITLE_ACCEPT = '.srt,.vtt'

let cueKeySeed = 0
function nextCueKey() {
  cueKeySeed += 1
  return `cue-${Date.now().toString(36)}-${cueKeySeed}`
}

function toCueDrafts(track: SubtitleTrackDto | null): CueDraft[] {
  if (!track) return []
  return track.cues.map((cue) => ({
    key: cue.id || nextCueKey(),
    start_ms: cue.start_ms,
    end_ms: cue.end_ms,
    text: cue.text,
    character_name: cue.character_name || '',
  }))
}

function describeError(error: unknown, fallback: string): string {
  const detail = (error as { response?: { data?: { detail?: unknown } } } | undefined)?.response?.data?.detail
  if (typeof detail === 'string' && detail) return detail
  return fallback
}

const AvWorkbench: React.FC = () => {
  const projectId = useProjectStore((s) => s.projectId)

  const [subtitleTracks, setSubtitleTracks] = useState<SubtitleTrackDto[]>([])
  const [selectedSubtitleTrackId, setSelectedSubtitleTrackId] = useState('')
  const [audioTracks, setAudioTracks] = useState<AudioTrackDto[]>([])
  const [shotInfos, setShotInfos] = useState<AvShotInfo[]>([])
  const [totalDurationMs, setTotalDurationMs] = useState(0)
  const [selectedTrackId, setSelectedTrackId] = useState('')
  const [cueDrafts, setCueDrafts] = useState<CueDraft[]>([])
  const [cueDirty, setCueDirty] = useState(false)
  const [undoStack, setUndoStack] = useState<UndoSnapshot<UndoAction>[]>([])
  const [warnings, setWarnings] = useState<AvWarning[]>([])
  const [_loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [lastAsset, setLastAsset] = useState<{ source_path: string; duration_ms: number; name: string } | null>(null)
  const [newTrackKind, setNewTrackKind] = useState<AudioTrackKind>('music')
  const [newTrackShotId, setNewTrackShotId] = useState('')
  const [previewStatus, setPreviewStatus] = useState<MixPreviewStatus | null>(null)
  const [playheadMs, setPlayheadMs] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [zoom, setZoom] = useState(1)

  const loadRequestRef = useRef(0)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const timelineAreaRef = useRef<HTMLDivElement | null>(null)
  const scrubbingRef = useRef(false)

  const selectedSubtitleTrack = useMemo(
    () => subtitleTracks.find((track) => track.id === selectedSubtitleTrackId) || subtitleTracks[0] || null,
    [subtitleTracks, selectedSubtitleTrackId],
  )
  const selectedTrack = useMemo(
    () => audioTracks.find((track) => track.id === selectedTrackId) || null,
    [audioTracks, selectedTrackId],
  )
  const pxPerSecond = [24, 40, 64, 100, 160][zoom] || 40
  const clips = useMemo(() => buildTrackClips(audioTracks, totalDurationMs), [audioTracks, totalDurationMs])
  const trackGroups = useMemo(() => groupTracksByKind(audioTracks), [audioTracks])
  const timelineWidth = msToPx(totalDurationMs + TIMELINE_PADDING_MS, pxPerSecond)

  const localWarnings = useMemo(
    () =>
      computeLocalWarnings({
        tracks: audioTracks,
        clips,
        shotInfos,
        cues: selectedSubtitleTrack?.cues || [],
        totalDurationMs,
      }),
    [audioTracks, clips, shotInfos, selectedSubtitleTrack, totalDurationMs],
  )
  const mergedWarnings = useMemo(() => {
    const map = new Map<string, AvWarning>()
    for (const item of localWarnings) map.set(`local:${item.code}:${item.track_id || ''}:${item.message}`, item)
    for (const item of warnings) map.set(`server:${item.code}:${item.track_id || ''}:${item.message}`, item)
    return [...map.values()]
  }, [localWarnings, warnings])

  const reload = useCallback(async () => {
    if (!projectId) return
    const request = ++loadRequestRef.current
    setLoading(true)
    try {
      const [subtitlePayload, audioPayload] = await Promise.all([
        subtitleApi.list(projectId),
        audioTrackApi.list(projectId),
      ])
      if (request !== loadRequestRef.current) return
      setSubtitleTracks(subtitlePayload.tracks)
      setSelectedSubtitleTrackId((current) => {
        if (current && subtitlePayload.tracks.some((track) => track.id === current)) return current
        return subtitlePayload.tracks[0]?.id || ''
      })
      setAudioTracks(audioPayload.tracks)
      setShotInfos(audioPayload.shots)
      setTotalDurationMs(audioPayload.total_duration_ms)
      setSelectedTrackId((current) => {
        if (current && audioPayload.tracks.some((track) => track.id === current)) return current
        return audioPayload.tracks[0]?.id || ''
      })
      // 草稿跟随当前选中的字幕轨（无选中时回落到第一条）。
      const preferred =
        subtitlePayload.tracks.find((track) => track.id === selectedSubtitleTrackId) ||
        subtitlePayload.tracks[0] ||
        null
      setCueDrafts(toCueDrafts(preferred))
      setCueDirty(false)
      setUndoStack([])
    } catch (error) {
      message.error(describeError(error, '字幕/音频配置加载失败'))
    } finally {
      if (request === loadRequestRef.current) setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  const refreshServerWarnings = useCallback(async () => {
    if (!projectId) return
    try {
      const result = await audioTrackApi.analyze(projectId)
      setWarnings(result.warnings || [])
    } catch {
      setWarnings([])
    }
  }, [projectId])

  useEffect(() => {
    void reload().then(() => void refreshServerWarnings())
  }, [reload, refreshServerWarnings])

  useEffect(() => {
    const status = previewStatus?.status
    if (!projectId || status !== 'mixing') return
    const timer = window.setInterval(async () => {
      try {
        const next = await audioTrackApi.previewStatus(projectId)
        setPreviewStatus(next)
      } catch {
        /* 轮询失败静默重试 */
      }
    }, 1500)
    return () => window.clearInterval(timer)
  }, [projectId, previewStatus?.status])

  // 预览播放时用 rAF 把音频进度同步到播放头。
  useEffect(() => {
    if (!isPlaying) return
    let frame = 0
    const tick = () => {
      const audio = audioElementRef.current
      if (audio && !audio.paused) setPlayheadMs(audio.currentTime * 1000)
      frame = window.requestAnimationFrame(tick)
    }
    frame = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(frame)
  }, [isPlaying])

  const applyUndo = useCallback(
    async (action: UndoAction) => {
      if (action.kind === 'cues') {
        setCueDrafts(action.cues)
        setCueDirty(true)
      } else {
        try {
          if (!projectId) return
          const updated = await audioTrackApi.updateTrack(action.trackId, { project_id: projectId, ...action.params })
          setAudioTracks((current) => current.map((track) => (track.id === updated.id ? updated : track)))
        } catch (error) {
          message.error(describeError(error, '撤销失败，请手动恢复参数'))
        }
      }
    },
    [projectId],
  )

  const handleUndo = useCallback(async () => {
    const step = undoOnce(undoStack, (action) => action)
    setUndoStack(step.stack)
    if (step.result) {
      await applyUndo(step.result)
      message.info(`已撤销：${step.result.label}`)
    }
  }, [undoStack, applyUndo])

  const pushCueSnapshot = useCallback(
    (label: string) => {
      setUndoStack((stack) => pushUndo(stack, label, { kind: 'cues', label, cues: cueDrafts }))
    },
    [cueDrafts],
  )

  // --- 字幕轨操作 ---

  const selectSubtitleTrack = (trackId: string) => {
    if (cueDirty) void saveCues(true)
    setSelectedSubtitleTrackId(trackId)
    const track = subtitleTracks.find((item) => item.id === trackId) || null
    setCueDrafts(toCueDrafts(track))
    setCueDirty(false)
  }

  const createSubtitleTrack = async () => {
    if (!projectId) return
    try {
      const track = await subtitleApi.createTrack(projectId, {})
      setSubtitleTracks((current) => [...current, track])
      setSelectedSubtitleTrackId(track.id)
      setCueDrafts([])
      setCueDirty(false)
    } catch (error) {
      message.error(describeError(error, '新建字幕轨失败'))
    }
  }

  const updateSubtitleTrack = async (patch: Record<string, unknown>) => {
    if (!projectId || !selectedSubtitleTrack) return
    try {
      const updated = await subtitleApi.updateTrack(selectedSubtitleTrack.id, { project_id: projectId, ...patch })
      setSubtitleTracks((current) => current.map((track) => (track.id === updated.id ? updated : track)))
    } catch (error) {
      message.error(describeError(error, '字幕样式保存失败'))
    }
  }

  const deleteSubtitleTrack = async () => {
    if (!projectId || !selectedSubtitleTrack) return
    try {
      await subtitleApi.deleteTrack(selectedSubtitleTrack.id, projectId)
      const remaining = subtitleTracks.filter((track) => track.id !== selectedSubtitleTrack.id)
      setSubtitleTracks(remaining)
      setSelectedSubtitleTrackId(remaining[0]?.id || '')
      setCueDrafts(toCueDrafts(remaining[0] || null))
      setCueDirty(false)
    } catch (error) {
      message.error(describeError(error, '删除字幕轨失败'))
    }
  }

  const saveCues = useCallback(
    async (silent = false) => {
      if (!projectId || !selectedSubtitleTrack) return
      const payload = cueDrafts.map((cue) => ({
        start_ms: cue.start_ms,
        end_ms: cue.end_ms,
        text: cue.text,
        character_name: cue.character_name,
      }))
      for (const cue of cueDrafts) {
        const problem = validateCueDraft(cue)
        if (problem) {
          message.warning(`字幕校验未通过：${problem}`)
          return
        }
      }
      try {
        const updated = await subtitleApi.replaceCues(selectedSubtitleTrack.id, {
          project_id: projectId,
          cues: payload,
        })
        setSubtitleTracks((current) => current.map((track) => (track.id === updated.id ? updated : track)))
        setCueDrafts(toCueDrafts(updated))
        setCueDirty(false)
        if (!silent) message.success('字幕已保存')
      } catch (error) {
        message.error(describeError(error, '字幕保存失败'))
      }
    },
    [projectId, selectedSubtitleTrack, cueDrafts],
  )

  const importSubtitleFile = async (file: File) => {
    if (!projectId || !selectedSubtitleTrack) {
      message.warning('请先创建并选中一条字幕轨')
      return
    }
    const format = file.name.toLowerCase().endsWith('.vtt') ? 'vtt' : 'srt'
    try {
      const content = await file.text()
      const updated = await subtitleApi.importSubtitle(selectedSubtitleTrack.id, {
        project_id: projectId,
        format,
        content,
      })
      setSubtitleTracks((current) => current.map((track) => (track.id === updated.id ? updated : track)))
      setCueDrafts(toCueDrafts(updated))
      setCueDirty(false)
      message.success(`已导入 ${updated.cues.length} 条字幕`)
    } catch (error) {
      message.error(describeError(error, '字幕导入失败'))
    }
  }

  const exportSubtitle = (format: 'srt' | 'vtt') => {
    if (!projectId || !selectedSubtitleTrack) {
      message.warning('请先选中一条字幕轨')
      return
    }
    const anchor = document.createElement('a')
    anchor.href = subtitleApi.exportUrl(selectedSubtitleTrack.id, projectId, format)
    anchor.download = `subtitle-${selectedSubtitleTrack.name || selectedSubtitleTrack.id}.${format}`
    anchor.click()
  }

  const generateFromShots = async () => {
    if (!projectId || !selectedSubtitleTrack) {
      message.warning('请先创建并选中一条字幕轨')
      return
    }
    try {
      const updated = await subtitleApi.generateFromShots(selectedSubtitleTrack.id, projectId)
      setSubtitleTracks((current) => current.map((track) => (track.id === updated.id ? updated : track)))
      setCueDrafts(toCueDrafts(updated))
      setCueDirty(false)
      const overlapCount = updated.overlaps?.length || 0
      message.success(
        `已按镜头对白生成 ${updated.cues.length} 条字幕${overlapCount ? `（${overlapCount} 处时间重叠，请检查）` : ''}`,
      )
    } catch (error) {
      message.error(describeError(error, '自动生成字幕失败'))
    }
  }

  // --- 字幕条目编辑 ---

  // 结构性修改（时间等失焦提交）压撤销快照；文本输入只更新草稿，避免逐字压栈。
  const updateCue = (key: string, patch: Partial<CueDraft>) => {
    setCueDrafts((current) => current.map((cue) => (cue.key === key ? { ...cue, ...patch } : cue)))
    setCueDirty(true)
  }

  const commitCueField = (key: string, patch: Partial<CueDraft>, label: string) => {
    pushCueSnapshot(label)
    setCueDrafts((current) => current.map((cue) => (cue.key === key ? { ...cue, ...patch } : cue)))
    setCueDirty(true)
  }

  const addCue = () => {
    const base = cueDrafts.length ? cueDrafts[cueDrafts.length - 1].end_ms : playheadMs
    pushCueSnapshot('新增字幕')
    setCueDrafts((current) => [
      ...current,
      { key: nextCueKey(), start_ms: base, end_ms: base + 1500, text: '', character_name: '' },
    ])
    setCueDirty(true)
  }

  const removeCue = (key: string) => {
    pushCueSnapshot('删除字幕')
    setCueDrafts((current) => current.filter((cue) => cue.key !== key))
    setCueDirty(true)
  }

  // --- 音频轨操作 ---

  const uploadAsset = async (file: File) => {
    if (!projectId) return
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const asset = await audioTrackApi.upload(projectId, formData)
      setLastAsset({ source_path: asset.source_path, duration_ms: asset.duration_ms, name: file.name })
      message.success(`素材已上传（${(asset.duration_ms / 1000).toFixed(1)}s），点击「添加轨道」使用`)
    } catch (error) {
      message.error(describeError(error, '音频素材上传失败'))
    } finally {
      setUploading(false)
    }
  }

  const addTrack = async () => {
    if (!projectId) return
    const payload: AudioTrackPayload = { kind: newTrackKind }
    if (newTrackKind === 'dialogue') {
      if (!newTrackShotId) {
        message.warning('对白轨需要选择绑定镜头')
        return
      }
      payload.shot_id = newTrackShotId
    } else {
      if (!lastAsset) {
        message.warning('请先上传音频素材')
        return
      }
      payload.source_path = lastAsset.source_path
    }
    try {
      const track = await audioTrackApi.createTrack(projectId, payload)
      setAudioTracks((current) => [...current, track])
      setSelectedTrackId(track.id)
      void refreshServerWarnings()
    } catch (error) {
      message.error(describeError(error, '添加轨道失败'))
    }
  }

  const updateTrack = async (patch: AudioTrackPayload, label: string) => {
    if (!projectId || !selectedTrack) return
    const previous: Record<string, unknown> = {}
    for (const key of Object.keys(patch)) {
      ;(previous as Record<string, unknown>)[key] = (selectedTrack as unknown as Record<string, unknown>)[key]
    }
    setUndoStack((stack) =>
      pushUndo(stack, label, { kind: 'track', label, trackId: selectedTrack.id, params: previous }),
    )
    try {
      const updated = await audioTrackApi.updateTrack(selectedTrack.id, { project_id: projectId, ...patch })
      setAudioTracks((current) => current.map((track) => (track.id === updated.id ? updated : track)))
    } catch (error) {
      message.error(describeError(error, '轨道参数保存失败'))
    }
  }

  const deleteTrack = async (trackId: string) => {
    if (!projectId) return
    try {
      await audioTrackApi.deleteTrack(trackId, projectId)
      setAudioTracks((current) => current.filter((track) => track.id !== trackId))
      if (selectedTrackId === trackId) setSelectedTrackId('')
      void refreshServerWarnings()
    } catch (error) {
      message.error(describeError(error, '删除轨道失败'))
    }
  }

  // --- 预览 ---

  const startPreview = async (scope: 'full' | 'shot') => {
    if (!projectId) return
    if (cueDirty) await saveCues(true)
    try {
      await audioTrackApi.startPreview(projectId, { scope, shot_id: scope === 'shot' ? newTrackShotId : '' })
      setPreviewStatus({ project_id: projectId, status: 'mixing', scope, progress: 0 })
    } catch (error) {
      message.error(describeError(error, '预览任务启动失败'))
    }
  }

  const previewUrl =
    previewStatus?.status === 'completed' && previewStatus.audio_path ? toOutputUrl(previewStatus.audio_path) : null

  const togglePlay = () => {
    const audio = audioElementRef.current
    if (!audio || !previewUrl) {
      message.warning('请先生成混音预览')
      return
    }
    if (audio.paused) {
      void audio.play()
      setIsPlaying(true)
    } else {
      audio.pause()
      setIsPlaying(false)
    }
  }

  // --- 播放头拖动 ---

  const setPlayheadFromClientX = useCallback(
    (clientX: number) => {
      const area = timelineAreaRef.current
      if (!area) return
      const bounds = area.getBoundingClientRect()
      const rawMs = ((clientX - bounds.left + area.scrollLeft) / pxPerSecond) * 1000
      const snapped = snapMs(
        clampMs(Math.max(0, Math.round(rawMs)), 0, totalDurationMs),
        snapCandidates(shotInfos, clips, selectedSubtitleTrack?.cues || [], Math.max(0, Math.round(rawMs))),
      )
      setPlayheadMs(snapped)
      const audio = audioElementRef.current
      if (audio && previewUrl) audio.currentTime = snapped / 1000
    },
    [pxPerSecond, totalDurationMs, shotInfos, clips, selectedSubtitleTrack, previewUrl],
  )

  useEffect(() => {
    const onMove = (event: MouseEvent) => {
      if (scrubbingRef.current) setPlayheadFromClientX(event.clientX)
    }
    const onUp = () => {
      scrubbingRef.current = false
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [setPlayheadFromClientX])

  const activeCue = cueAtPlayhead(cueDrafts, playheadMs)
  const errorWarnings = mergedWarnings.filter((item) => item.level === 'error')
  const otherWarnings = mergedWarnings.filter((item) => item.level !== 'error')

  const secondsField = (valueMs: number) => (valueMs / 1000).toFixed(3)
  const parseSecondsField = (text: string, fallback: number) => {
    const parsed = Number(text)
    if (!Number.isFinite(parsed) || parsed < 0) return fallback
    return Math.round(parsed * 1000)
  }

  return (
    <div className="av-workbench">
      <div className="av-toolbar">
        <div className="av-toolbar-group">
          <Select
            size="small"
            style={{ width: 168 }}
            value={selectedSubtitleTrack?.id || ''}
            onChange={selectSubtitleTrack}
            placeholder="字幕轨"
            options={subtitleTracks.map((track) => ({ value: track.id, label: track.name }))}
          />
          <Button size="small" onClick={() => void createSubtitleTrack()}>
            新建字幕轨
          </Button>
          <Button size="small" danger disabled={!selectedSubtitleTrack} onClick={() => void deleteSubtitleTrack()}>
            删除
          </Button>
        </div>
        <div className="av-toolbar-group">
          <Button
            size="small"
            icon={<UploadOutlined />}
            disabled={!selectedSubtitleTrack}
            onClick={() => document.getElementById('av-subtitle-import')?.click()}
          >
            导入 SRT/VTT
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            disabled={!selectedSubtitleTrack}
            onClick={() => exportSubtitle('srt')}
          >
            导出 SRT
          </Button>
          <Button
            size="small"
            icon={<DownloadOutlined />}
            disabled={!selectedSubtitleTrack}
            onClick={() => exportSubtitle('vtt')}
          >
            导出 VTT
          </Button>
          <Button size="small" disabled={!selectedSubtitleTrack} onClick={() => void generateFromShots()}>
            按镜头对白生成
          </Button>
        </div>
        <div className="av-toolbar-group">
          <label className="av-file-button">
            <UploadOutlined /> 上传音频素材
            <input
              type="file"
              accept={AUDIO_ACCEPT}
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void uploadAsset(file)
                event.target.value = ''
              }}
            />
          </label>
          <Select
            size="small"
            style={{ width: 108 }}
            value={newTrackKind}
            onChange={setNewTrackKind}
            options={[
              { value: 'music', label: '背景音乐' },
              { value: 'ambient', label: '环境音' },
              { value: 'sfx', label: '音效' },
              { value: 'dialogue', label: '对白' },
            ]}
          />
          {newTrackKind === 'dialogue' ? (
            <Select
              size="small"
              style={{ width: 140 }}
              value={newTrackShotId || undefined}
              onChange={setNewTrackShotId}
              placeholder="绑定镜头"
              options={shotInfos.map((info) => ({
                value: info.id,
                label: `镜头 ${info.sequence}${info.has_tts ? '' : '（无配音）'}`,
              }))}
            />
          ) : (
            <span className="av-toolbar-hint" title={lastAsset?.name || ''}>
              {lastAsset ? `素材 ${(lastAsset.duration_ms / 1000).toFixed(1)}s` : '未上传素材'}
            </span>
          )}
          <Button size="small" type="primary" loading={uploading} onClick={() => void addTrack()}>
            添加轨道
          </Button>
        </div>
        <div className="av-toolbar-group">
          <Button size="small" icon={<UndoOutlined />} disabled={!canUndo(undoStack)} onClick={() => void handleUndo()}>
            撤销
          </Button>
          <Button size="small" onClick={() => void refreshServerWarnings()}>
            重新检查
          </Button>
          <Select
            size="small"
            style={{ width: 84 }}
            value={zoom}
            onChange={setZoom}
            options={[
              { value: 0, label: '小' },
              { value: 1, label: '中' },
              { value: 2, label: '大' },
              { value: 3, label: '特大' },
              { value: 4, label: '最大' },
            ]}
          />
        </div>
      </div>

      <input
        id="av-subtitle-import"
        type="file"
        accept={SUBTITLE_ACCEPT}
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) void importSubtitleFile(file)
          event.target.value = ''
        }}
      />

      {(errorWarnings.length > 0 || otherWarnings.length > 0) && (
        <div className="av-warnings" aria-label="工作台告警">
          {errorWarnings.map((item, index) => (
            <span key={`e${index}`} className="av-warning error">
              {item.message}
            </span>
          ))}
          {otherWarnings.slice(0, 6).map((item, index) => (
            <span key={`w${index}`} className={`av-warning ${item.level}`}>
              {item.message}
            </span>
          ))}
        </div>
      )}

      <div className="av-layout">
        <div className="av-timeline-card">
          <div className="av-timeline-head">
            <span className="av-timecode">
              {formatTimelineMs(playheadMs)} / {formatTimelineMs(totalDurationMs)}
            </span>
            <div className="av-timeline-actions">
              <Button
                size="small"
                icon={<AudioOutlined />}
                loading={previewStatus?.status === 'mixing'}
                onClick={() => void startPreview('full')}
              >
                整片预览
              </Button>
              <Button
                size="small"
                disabled={!newTrackShotId}
                loading={previewStatus?.status === 'mixing'}
                onClick={() => void startPreview('shot')}
              >
                镜头预览
              </Button>
              <Button size="small" onClick={togglePlay}>
                {isPlaying ? '暂停' : '播放'}
              </Button>
            </div>
          </div>
          {previewStatus?.status === 'error' && (
            <div className="av-preview-error">{previewStatus.message || '混音预览失败'}</div>
          )}
          <div className="av-timeline-scroll" ref={timelineAreaRef}>
            <div className="av-timeline-inner" style={{ width: Math.max(timelineWidth, 320) }}>
              <div
                className="av-ruler"
                onMouseDown={(event) => {
                  scrubbingRef.current = true
                  setPlayheadFromClientX(event.clientX)
                }}
              >
                {Array.from(
                  { length: Math.ceil((totalDurationMs + TIMELINE_PADDING_MS) / 5000) + 1 },
                  (_, index) => index * 5000,
                ).map((ms) => (
                  <span key={ms} className="av-ruler-mark" style={{ left: msToPx(ms, pxPerSecond) }}>
                    {formatTimelineMs(ms)}
                  </span>
                ))}
              </div>
              <div className="av-lanes">
                <div className="av-lane av-lane-shots">
                  {shotInfos.map((info) => (
                    <div
                      key={info.id}
                      className={`av-shot-block${info.native_audio ? ' native' : ''}${info.has_tts ? ' tts' : ''}`}
                      style={{
                        left: msToPx(info.start_ms, pxPerSecond),
                        width: msToPx(Math.max(200, info.duration_ms - 40), pxPerSecond),
                      }}
                      title={`镜头 ${info.sequence} · ${formatTimelineMs(info.start_ms)}–${formatTimelineMs(info.end_ms)}${info.dialogue ? `\n对白：${info.dialogue}` : ''}`}
                    >
                      镜头 {info.sequence}
                    </div>
                  ))}
                </div>
                {trackGroups.map((group) => (
                  <div className="av-lane" key={group.kind} data-kind={group.kind}>
                    <span className="av-lane-label">{TRACK_KIND_LABELS[group.kind]}</span>
                    {group.items.map((track) => {
                      const clip = clips.find((item) => item.track.id === track.id)
                      if (!clip) return null
                      const selected = track.id === selectedTrackId
                      return (
                        <div
                          key={track.id}
                          className={`av-clip${selected ? ' selected' : ''}${clip.outOfRange ? ' invalid' : ''}${track.muted ? ' muted' : ''}`}
                          style={{
                            left: msToPx(clip.start_ms, pxPerSecond),
                            width: Math.max(14, msToPx(clip.duration_ms, pxPerSecond)),
                          }}
                          onClick={() => setSelectedTrackId(track.id)}
                          title={`${track.name} · ${formatTimelineMs(clip.start_ms)} 起共 ${(clip.duration_ms / 1000).toFixed(1)}s${track.muted ? '（静音）' : ''}`}
                        >
                          <span className="av-clip-name">{track.name || TRACK_KIND_LABELS[track.kind]}</span>
                          <span className="av-clip-meta">
                            {track.muted ? '静音' : `×${track.volume.toFixed(2)}`}
                            {track.loop ? ' · 循环' : ''}
                            {track.duck_amount_db < 0 ? ` · Duck ${track.duck_amount_db}dB` : ''}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                ))}
                <div className="av-lane av-lane-subtitle">
                  <span className="av-lane-label">字幕</span>
                  {cueDrafts.map((cue) => (
                    <div
                      key={cue.key}
                      className="av-cue-block"
                      style={{
                        left: msToPx(cue.start_ms, pxPerSecond),
                        width: Math.max(12, msToPx(Math.max(200, cue.end_ms - cue.start_ms), pxPerSecond)),
                      }}
                      title={`${formatTimelineMs(cue.start_ms)}–${formatTimelineMs(cue.end_ms)}\n${cue.text}`}
                    >
                      {cue.text.slice(0, 12)}
                    </div>
                  ))}
                </div>
                <div className="av-playhead" style={{ left: msToPx(playheadMs, pxPerSecond) }} />
                {activeCue && (
                  <div
                    className="av-playhead-caption"
                    style={{ left: clampMs(msToPx(playheadMs, pxPerSecond), 0, Math.max(0, timelineWidth - 240)) }}
                  >
                    {activeCue.text}
                  </div>
                )}
              </div>
            </div>
          </div>
          {previewUrl && <audio ref={audioElementRef} src={previewUrl} onEnded={() => setIsPlaying(false)} hidden />}
        </div>

        <div className="av-inspector">
          <section className="av-inspector-section">
            <h4>轨道参数{selectedTrack ? ` · ${selectedTrack.name || TRACK_KIND_LABELS[selectedTrack.kind]}` : ''}</h4>
            {!selectedTrack ? (
              <p className="av-inspector-empty">点击时间线上的轨道块进行调整。</p>
            ) : (
              <div className="av-param-grid">
                <label>
                  音量 <em>{selectedTrack.volume.toFixed(2)}</em>
                </label>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  defaultValue={selectedTrack.volume}
                  onMouseUp={(event) =>
                    void updateTrack({ volume: Number((event.target as HTMLInputElement).value) }, '调整音量')
                  }
                  onTouchEnd={(event) =>
                    void updateTrack({ volume: Number((event.target as HTMLInputElement).value) }, '调整音量')
                  }
                />
                <label>
                  声像 <em>{selectedTrack.pan.toFixed(2)}</em>
                </label>
                <input
                  type="range"
                  min={-1}
                  max={1}
                  step={0.05}
                  defaultValue={selectedTrack.pan}
                  onMouseUp={(event) =>
                    void updateTrack({ pan: Number((event.target as HTMLInputElement).value) }, '调整声像')
                  }
                  onTouchEnd={(event) =>
                    void updateTrack({ pan: Number((event.target as HTMLInputElement).value) }, '调整声像')
                  }
                />
                <label>起点 (秒)</label>
                <Input
                  size="small"
                  defaultValue={secondsField(selectedTrack.start_ms)}
                  onBlur={(event) =>
                    void updateTrack(
                      { start_ms: parseSecondsField(event.target.value, selectedTrack.start_ms) },
                      '调整起点',
                    )
                  }
                />
                <label>淡入 (秒)</label>
                <Input
                  size="small"
                  defaultValue={secondsField(selectedTrack.fade_in_ms)}
                  onBlur={(event) =>
                    void updateTrack(
                      { fade_in_ms: parseSecondsField(event.target.value, selectedTrack.fade_in_ms) },
                      '调整淡入',
                    )
                  }
                />
                <label>淡出 (秒)</label>
                <Input
                  size="small"
                  defaultValue={secondsField(selectedTrack.fade_out_ms)}
                  onBlur={(event) =>
                    void updateTrack(
                      { fade_out_ms: parseSecondsField(event.target.value, selectedTrack.fade_out_ms) },
                      '调整淡出',
                    )
                  }
                />
                <label>延迟 (秒)</label>
                <Input
                  size="small"
                  defaultValue={secondsField(selectedTrack.delay_ms)}
                  onBlur={(event) =>
                    void updateTrack(
                      { delay_ms: parseSecondsField(event.target.value, selectedTrack.delay_ms) },
                      '调整延迟',
                    )
                  }
                />
                <label>裁剪起点 (秒)</label>
                <Input
                  size="small"
                  defaultValue={secondsField(selectedTrack.trim_start_ms)}
                  onBlur={(event) =>
                    void updateTrack(
                      { trim_start_ms: parseSecondsField(event.target.value, selectedTrack.trim_start_ms) },
                      '调整裁剪',
                    )
                  }
                />
                <label>裁剪结尾 (秒)</label>
                <Input
                  size="small"
                  defaultValue={secondsField(selectedTrack.trim_end_ms)}
                  onBlur={(event) =>
                    void updateTrack(
                      { trim_end_ms: parseSecondsField(event.target.value, selectedTrack.trim_end_ms) },
                      '调整裁剪',
                    )
                  }
                />
                <label>循环</label>
                <Switch
                  size="small"
                  checked={selectedTrack.loop}
                  onChange={(checked) => void updateTrack({ loop: checked }, '切换循环')}
                />
                <label>静音</label>
                <Switch
                  size="small"
                  checked={selectedTrack.muted}
                  onChange={(checked) => void updateTrack({ muted: checked }, '切换静音')}
                />
                {selectedTrack.kind !== 'dialogue' && (
                  <>
                    <label>Ducking (dB)</label>
                    <Input
                      size="small"
                      defaultValue={String(selectedTrack.duck_amount_db)}
                      onBlur={(event) => {
                        const parsed = Number(event.target.value)
                        void updateTrack(
                          { duck_amount_db: Math.max(-48, Math.min(0, Number.isFinite(parsed) ? parsed : 0)) },
                          '调整 Ducking',
                        )
                      }}
                    />
                    <label>起控/释控 (ms)</label>
                    <div className="av-param-pair">
                      <Input
                        size="small"
                        defaultValue={String(selectedTrack.duck_attack_ms)}
                        onBlur={(event) =>
                          void updateTrack(
                            { duck_attack_ms: parseSecondsField(event.target.value, selectedTrack.duck_attack_ms) },
                            '调整起控',
                          )
                        }
                      />
                      <Input
                        size="small"
                        defaultValue={String(selectedTrack.duck_release_ms)}
                        onBlur={(event) =>
                          void updateTrack(
                            { duck_release_ms: parseSecondsField(event.target.value, selectedTrack.duck_release_ms) },
                            '调整释控',
                          )
                        }
                      />
                    </div>
                  </>
                )}
                <Button
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => void deleteTrack(selectedTrack.id)}
                >
                  删除轨道
                </Button>
              </div>
            )}
          </section>

          <section className="av-inspector-section">
            <h4>字幕样式{selectedSubtitleTrack ? ` · ${selectedSubtitleTrack.name}` : ''}</h4>
            {!selectedSubtitleTrack ? (
              <p className="av-inspector-empty">请新建一条字幕轨。</p>
            ) : (
              <div className="av-param-grid">
                <label>启用</label>
                <Switch
                  size="small"
                  checked={selectedSubtitleTrack.enabled}
                  onChange={(checked) => void updateSubtitleTrack({ enabled: checked })}
                />
                <label>烧录进画面</label>
                <Switch
                  size="small"
                  checked={selectedSubtitleTrack.burn_in}
                  onChange={(checked) => void updateSubtitleTrack({ burn_in: checked })}
                />
                <label>位置</label>
                <Select
                  size="small"
                  value={selectedSubtitleTrack.position}
                  onChange={(value) => void updateSubtitleTrack({ position: value })}
                  options={[
                    { value: 'bottom', label: '底部' },
                    { value: 'middle', label: '中部' },
                    { value: 'top', label: '顶部' },
                  ]}
                />
                <label>字号</label>
                <Input
                  size="small"
                  defaultValue={String(selectedSubtitleTrack.font_size)}
                  onBlur={(event) => {
                    const parsed = Number(event.target.value)
                    if (Number.isFinite(parsed))
                      void updateSubtitleTrack({ font_size: Math.max(8, Math.min(200, Math.round(parsed))) })
                  }}
                />
                <label>描边宽度</label>
                <Input
                  size="small"
                  defaultValue={String(selectedSubtitleTrack.outline_width)}
                  onBlur={(event) => {
                    const parsed = Number(event.target.value)
                    if (Number.isFinite(parsed))
                      void updateSubtitleTrack({ outline_width: Math.max(0, Math.min(20, Math.round(parsed))) })
                  }}
                />
                <label>安全区边距</label>
                <Input
                  size="small"
                  defaultValue={String(selectedSubtitleTrack.safe_margin)}
                  onBlur={(event) => {
                    const parsed = Number(event.target.value)
                    if (Number.isFinite(parsed))
                      void updateSubtitleTrack({ safe_margin: Math.max(0, Math.min(400, Math.round(parsed))) })
                  }}
                />
                <label>文字颜色</label>
                <Input
                  size="small"
                  defaultValue={selectedSubtitleTrack.primary_color}
                  onBlur={(event) =>
                    /^#[0-9A-Fa-f]{6}$/.test(event.target.value.trim()) &&
                    void updateSubtitleTrack({ primary_color: event.target.value.trim().toUpperCase() })
                  }
                />
                <label>描边颜色</label>
                <Input
                  size="small"
                  defaultValue={selectedSubtitleTrack.outline_color}
                  onBlur={(event) =>
                    /^#[0-9A-Fa-f]{6}$/.test(event.target.value.trim()) &&
                    void updateSubtitleTrack({ outline_color: event.target.value.trim().toUpperCase() })
                  }
                />
              </div>
            )}
          </section>
        </div>
      </div>

      <div className="av-cue-editor">
        <div className="av-cue-editor-head">
          <h4>字幕条目（{cueDrafts.length}）</h4>
          <div>
            <Button size="small" onClick={addCue}>
              新增
            </Button>{' '}
            <Button
              size="small"
              type="primary"
              disabled={!cueDirty || !selectedSubtitleTrack}
              onClick={() => void saveCues()}
            >
              保存字幕
            </Button>
            {cueDirty && <em className="av-dirty-flag">有未保存修改</em>}
          </div>
        </div>
        <div className="av-cue-table" role="table">
          <div className="av-cue-row av-cue-header" role="row">
            <span>开始 (秒)</span>
            <span>结束 (秒)</span>
            <span>文本</span>
            <span>角色</span>
            <span />
          </div>
          {cueDrafts.map((cue, index) => {
            const problem = validateCueDraft(cue)
            return (
              <div key={cue.key} className={`av-cue-row${problem ? ' invalid' : ''}`} role="row">
                <Input
                  size="small"
                  aria-label={`第 ${index + 1} 条开始时间`}
                  defaultValue={secondsField(cue.start_ms)}
                  onBlur={(event) =>
                    commitCueField(
                      cue.key,
                      { start_ms: parseSecondsField(event.target.value, cue.start_ms) },
                      '调整开始时间',
                    )
                  }
                />
                <Input
                  size="small"
                  aria-label={`第 ${index + 1} 条结束时间`}
                  defaultValue={secondsField(cue.end_ms)}
                  onBlur={(event) =>
                    commitCueField(
                      cue.key,
                      { end_ms: parseSecondsField(event.target.value, cue.end_ms) },
                      '调整结束时间',
                    )
                  }
                />
                <Input
                  size="small"
                  aria-label={`第 ${index + 1} 条文本`}
                  defaultValue={cue.text}
                  onChange={(event) => updateCue(cue.key, { text: event.target.value })}
                />
                <Input
                  size="small"
                  aria-label={`第 ${index + 1} 条角色`}
                  defaultValue={cue.character_name}
                  onChange={(event) => updateCue(cue.key, { character_name: event.target.value })}
                />
                <Button
                  size="small"
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => removeCue(cue.key)}
                  aria-label={`删除第 ${index + 1} 条`}
                />
              </div>
            )
          })}
          {!cueDrafts.length && <p className="av-cue-empty">还没有字幕：点「按镜头对白生成」或导入 SRT/VTT。</p>}
        </div>
      </div>
    </div>
  )
}

export default AvWorkbench
