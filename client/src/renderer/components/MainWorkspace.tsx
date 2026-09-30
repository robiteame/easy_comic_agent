import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import Button from 'antd/es/button'
import Input from 'antd/es/input'
import message from 'antd/es/message'
import Modal from 'antd/es/modal'
import Segmented from 'antd/es/segmented'
import Select from 'antd/es/select'
import Tooltip from 'antd/es/tooltip'
import {
  BulbOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  DragOutlined,
  EditOutlined,
  ReloadOutlined,
  SaveOutlined,
  SendOutlined,
  UploadOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
} from '@ant-design/icons'
import { useShotStore } from '../stores/shotStore'
import { useProjectStore } from '../stores/projectStore'
import { useTaskStore } from '../stores/taskStore'
import { assetApi, createWebSocket, projectApi, regenerationQueueApi, renderApi, scriptApi, settingsApi, shotApi, toOutputUrl } from '../services/api'
import type { RegenerationQueueStage } from '../services/api'
import {
  beginProjectNavigationIntent,
  currentProjectNavigationIntent,
  requestProjectNavigation,
} from '../services/projectNavigationGuard'
import { isCurrentProjectAsyncSnapshot, isLatestResourceResponse } from '../services/asyncGuard'
import { STYLE_DESCRIPTIONS, STYLE_OPTIONS } from '../constants/styleTemplates'
import { PROJECTS_REFRESHED_EVENT, STYLE_TEMPLATES_UPDATED_EVENT } from '../constants/events'
import {
  getWorkspacePanelAriaProps,
  getWorkspaceTabAriaProps,
  resolveWorkspaceTabFocus,
} from './workspaceTabs'

const AvWorkbench = React.lazy(() => import('./AvWorkbench'))
import BudgetSummaryPanel from './BudgetSummaryPanel'
import QualityReviewPanel from './QualityReviewPanel'
import { qualityBadgeFor, verdictMeta } from './qualityReviewModel'
import { notifyBudgetBlocked, notifyBudgetWarning, notifyProviderBlocked, useTaskEstimateGate } from './TaskEstimateModal'

const { TextArea } = Input
const PARSE_SCRIPT_EVENT = 'pipeline:parse-script'
const OPEN_CREATE_PROJECT_EVENT = 'workspace:open-create-project'
const OPEN_SHOT_CONFIG_EVENT = 'workspace:open-shot-config'
const WORKSPACE_NAVIGATE_EVENT = 'workspace:navigate'

type StyleOption = { value: string; label: string; keywords?: string; custom?: boolean }

const STEP_LABELS: Record<string, string> = {
  start: '开始',
  parse_script: '剧本解析',
  generate_storyboard: '分镜生成',
  wait_asset_confirm: '等待素材确认',
  generate_storyboard_images: '故事板生成',
  wait_storyboard_approval: '等待故事板审核',
  phase2_start: '进入第二阶段',
  generate_voice: '配音生成',
  generate_seedance_video: 'Seedance 视频',
  compose_video: '视频合成',
  quality_check: '结构检查（仅结构，非质量认证）',
  rendering: '导出渲染',
}

const WORKSPACE_TABS = [
  { id: 'script', label: '剧本编辑' },
  { id: 'assets', label: '角色场景资产' },
  { id: 'storyboard', label: '故事板预览' },
  { id: 'review', label: '分镜审核' },
  { id: 'av', label: '字幕与音频' },
  { id: 'video', label: '成片预览' },
] as const

type WorkspaceTab = typeof WORKSPACE_TABS[number]['id']

type ProjectOperation = {
  key: string
  token: number
  projectId: string | null
  projectEpoch: number
  navigationIntent: number
}

function normalizeShot(shot: any) {
  return {
    id: shot.id || shot.shot_id || '',
    project_id: shot.project_id || '',
    sequence: Number(shot.sequence || 0),
    shot_type: shot.shot_type || 'medium',
    scene_description: shot.scene_description || '',
    character_action: shot.character_action || '',
    dialogue: shot.dialogue || '',
    camera_angle: shot.camera_angle || '正面',
    camera_movement: shot.camera_movement || '静止',
    duration: Number(shot.duration || 3),
    emotion: shot.emotion || 'neutral',
    transition: shot.transition || 'cut',
    visual_notes: shot.visual_notes || '',
    image_path: shot.image_path || '',
    storyboard_path: shot.storyboard_path || '',
    video_path: shot.video_path || '',
    audio_path: shot.audio_path || '',
    status: shot.status || 'pending',
    storyboard_status: shot.storyboard_status || 'pending',
    version: Number(shot.version || 1),
    confirmed: Boolean(shot.confirmed),
    media_stale: Boolean(shot.media_stale),
    characters_in_scene: Array.isArray(shot.characters_in_scene) ? shot.characters_in_scene : [],
    scene_asset_id: shot.scene_asset_id || '',
    character_asset_ids: Array.isArray(shot.character_asset_ids) ? shot.character_asset_ids : [],
    scene_group_id: shot.scene_group_id || '',
    consistency_context: shot.consistency_context || '',
    reference_weights: shot.reference_weights || {},
    continuity_profile: shot.continuity_profile || {},
    continuity_reference_path: shot.continuity_reference_path || '',
    pose_reference_path: shot.pose_reference_path || '',
    depth_reference_path: shot.depth_reference_path || '',
    last_frame_path: shot.last_frame_path || '',
  }
}

function getStepLabel(step?: string) {
  if (!step) return '处理中'
  return STEP_LABELS[step] || '处理中'
}

const MainWorkspace: React.FC = () => {
  const {
    shots,
    selectedShotId,
    selectShot,
    setShots,
    setGenerating,
    setProgress,
    isGenerating,
    updateShot,
    applyServerShotUpdate,
    awaitingStoryboardConfirm,
    setAwaitingStoryboardConfirm,
    videoPath,
    setVideoPath,
    appendLog,
    clearLogs,
    currentStep,
  } = useShotStore()

  const {
    projectId,
    parentProjectTitle,
    projectType,
    title,
    setProject,
    style,
    platform,
    outputFormat,
    resolution,
    runMode,
  } = useProjectStore()

  // 提交前的成本估算闸门：所有会触发付费任务的入口都要先过一遍它。
  const estimateGate = useTaskEstimateGate()

  const [script, setScript] = useState('')
  const [newProjectTitle, setNewProjectTitle] = useState('')
  const [newEpisodeTitle, setNewEpisodeTitle] = useState('第 1 集')
  const [episodeTitleDraft, setEpisodeTitleDraft] = useState(title)
  const [showCreatePanel, setShowCreatePanel] = useState(false)
  const [creatingProject, setCreatingProject] = useState(false)
  const [loading, setLoading] = useState(false)
  const [autoWriting, setAutoWriting] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [composing, setComposing] = useState(false)
  const [generatingStoryboard, setGeneratingStoryboard] = useState(false)
  const [queueSelectedIds, setQueueSelectedIds] = useState<string[]>([])
  const [queueStages, setQueueStages] = useState<RegenerationQueueStage[]>(['storyboard'])
  const [queuePriority, setQueuePriority] = useState(0)
  const [queueConcurrency, setQueueConcurrency] = useState(1)
  const [queueOrder, setQueueOrder] = useState<'shot' | 'sequence' | 'reverse'>('sequence')
  const [queueReuseAudio, setQueueReuseAudio] = useState(true)
  const [queueResumeMissing, setQueueResumeMissing] = useState(false)
  const [queueForceConfirmed, setQueueForceConfirmed] = useState(false)
  const [queueVersion, setQueueVersion] = useState(0)
  const [queueSubmitting, setQueueSubmitting] = useState(false)
  const [assetBoard, setAssetBoard] = useState<{ characters: any[]; scenes: any[] } | null>(null)
  const [assetBoardReady, setAssetBoardReady] = useState(false)
  const [assetTab, setAssetTab] = useState<'characters' | 'scenes'>('characters')
  const [styleTemplates, setStyleTemplates] = useState<StyleOption[]>(STYLE_OPTIONS)
  // 后端回传的实际生效风格（requested/effective/source），与选择器分离展示。
  const [effectiveStyleMeta, setEffectiveStyleMeta] = useState<{
    requested_style?: string
    effective_style?: string
    style_source?: string
  } | null>(null)
  const [rebuildingAssets, setRebuildingAssets] = useState(false)
  const [editingAssetId, setEditingAssetId] = useState<string | null>(null)
  const [assetDraft, setAssetDraft] = useState<Record<string, any>>({})
  const [savingAsset, setSavingAsset] = useState(false)
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>('script')
  const [previewMode, setPreviewMode] = useState<'shot' | 'video'>('shot')
  const [previewScale, setPreviewScale] = useState(1)
  const [previewOffset, setPreviewOffset] = useState({ x: 0, y: 0 })
  const [dragStart, setDragStart] = useState<{ x: number; y: number; ox: number; oy: number } | null>(null)
  const [imagePreview, setImagePreview] = useState<{ url: string; title: string } | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const wsProjectIdRef = useRef<string | null>(null)
  const wsProjectEpochRef = useRef(0)
  const wsEpochRef = useRef(0)
  const wsReconnectTimerRef = useRef<number | null>(null)
  const wsReconnectAttemptRef = useRef(0)
  const wsClosedByUserRef = useRef(false)
  const projectEpochRef = useRef(0)
  const activeProjectIdRef = useRef<string | null>(projectId)
  const activatedProjectRef = useRef<{ projectId: string; projectEpoch: number } | null>(null)
  const operationTokensRef = useRef(new Map<string, number>())
  const projectCreationPromiseRef = useRef<Promise<any> | null>(null)
  const shotLoadRequestRef = useRef(0)
  const assetLoadRequestRef = useRef(0)
  const shotMutationRef = useRef(0)
  const assetMutationRef = useRef(0)
  const styleTemplateRequestRef = useRef(0)
  const mountedRef = useRef(false)
  const initializingProjectRef = useRef(false)
  const initializingProjectIdRef = useRef<string | null>(null)
  const initialScriptRef = useRef('')
  const uploadInputRef = useRef<HTMLInputElement | null>(null)
  const workspaceTabRefs = useRef(new Map<WorkspaceTab, HTMLButtonElement>())
  const generateRef = useRef<() => Promise<void>>(async () => {})
  const awaitingRef = useRef(awaitingStoryboardConfirm)
  const stepRef = useRef(currentStep)
  const pollingRef = useRef(false)

  const selectedShot = useMemo(
    () => shots.find((s) => s.id === selectedShotId) || shots[0],
    [shots, selectedShotId],
  )

  const hasStoryboardShots = useMemo(
    () => shots.some((shot) => Boolean(shot.storyboard_path || shot.image_path)),
    [shots],
  )
  const storyboardReviewVisible = hasStoryboardShots && previewMode === 'shot' && !videoPath
  const approvedShotCount = useMemo(
    () => shots.filter((shot) => shot.confirmed).length,
    [shots],
  )
  const selectedShotVideoUrl = selectedShot?.video_path ? toOutputUrl(selectedShot.video_path) : null
  const allShotVideosReady = useMemo(
    () => shots.length > 0 && shots.every((shot) => Boolean(shot.video_path)),
    [shots],
  )

  useEffect(() => {
    mountedRef.current = true
    let previousShots = useShotStore.getState().shots
    const unsubscribe = useShotStore.subscribe((state) => {
      if (state.shots !== previousShots) {
        previousShots = state.shots
        shotMutationRef.current += 1
      }
    })
    return () => {
      mountedRef.current = false
      operationTokensRef.current.clear()
      shotLoadRequestRef.current += 1
      assetLoadRequestRef.current += 1
      styleTemplateRequestRef.current += 1
      unsubscribe()
    }
  }, [])

  useEffect(() => {
    const navigateWorkspace = (event: Event) => {
      const detail = (event as CustomEvent<{ tab?: WorkspaceTab; previewMode?: 'shot' | 'video' }>).detail || {}
      const nextTab = detail.tab
      if (nextTab && WORKSPACE_TABS.some((tab) => tab.id === nextTab)) {
        setWorkspaceTab(nextTab)
      }
      if (detail.previewMode) {
        setPreviewMode(detail.previewMode)
      } else if (nextTab === 'storyboard' || nextTab === 'review') {
        setPreviewMode('shot')
      } else if (nextTab === 'video') {
        setPreviewMode('video')
      }
    }

    window.addEventListener(WORKSPACE_NAVIGATE_EVENT, navigateWorkspace)
    return () => window.removeEventListener(WORKSPACE_NAVIGATE_EVENT, navigateWorkspace)
  }, [])

  useEffect(() => {
    const openShot = (event: Event) => {
      const detail = (event as CustomEvent<{ projectId?: string; shotId?: string }>).detail || {}
      if (!detail.shotId || (detail.projectId && detail.projectId !== useProjectStore.getState().projectId)) return
      openShotConfig(detail.shotId)
    }
    window.addEventListener('workspace:open-shot', openShot)
    return () => window.removeEventListener('workspace:open-shot', openShot)
  }, [projectId])

  const isCurrentProject = (pid: string, epoch = projectEpochRef.current) =>
    activeProjectIdRef.current === pid && projectEpochRef.current === epoch && useProjectStore.getState().projectId === pid

  const beginOperation = (key: string, operationProjectId: string | null = projectId): ProjectOperation => {
    const token = (operationTokensRef.current.get(key) || 0) + 1
    operationTokensRef.current.set(key, token)
    return {
      key,
      token,
      projectId: operationProjectId,
      projectEpoch: projectEpochRef.current,
      navigationIntent: currentProjectNavigationIntent(),
    }
  }

  const isLatestOperation = (operation: ProjectOperation) =>
    mountedRef.current && operationTokensRef.current.get(operation.key) === operation.token

  const isCurrentOperation = (operation: ProjectOperation) =>
    activeProjectIdRef.current === useProjectStore.getState().projectId &&
    isCurrentProjectAsyncSnapshot(
      {
        projectId: operation.projectId,
        projectEpoch: operation.projectEpoch,
        operationToken: operation.token,
      },
      {
        projectId: activeProjectIdRef.current,
        projectEpoch: projectEpochRef.current,
        operationToken: operationTokensRef.current.get(operation.key) || 0,
      },
      mountedRef.current,
    )

  const rebaseOperation = (
    operation: ProjectOperation,
    operationProjectId: string,
    projectEpoch: number,
  ): ProjectOperation => ({ ...operation, projectId: operationProjectId, projectEpoch })

  const bindOperationToProject = (
    operation: ProjectOperation,
    context: { projectId: string; projectEpoch: number },
  ): ProjectOperation | null => {
    if (
      !isLatestOperation(operation) ||
      operation.navigationIntent !== currentProjectNavigationIntent()
    ) return null
    const bound = operation.projectId === null
      ? rebaseOperation(operation, context.projectId, context.projectEpoch)
      : operation
    return bound.projectId === context.projectId &&
      bound.projectEpoch === context.projectEpoch &&
      isCurrentOperation(bound)
      ? bound
      : null
  }

  const replaceShots = (nextShots: any[]) => {
    const normalized = nextShots.map(normalizeShot).filter((s) => s.id)
    setShots(normalized)
    const currentSelectedShotId = useShotStore.getState().selectedShotId
    const stillExists = normalized.some((s) => s.id === currentSelectedShotId)
    if (!stillExists && normalized[0]?.id) {
      selectShot(normalized[0].id)
    } else if (!normalized.length) {
      selectShot(null)
    }
  }

  const loadProjectShots = async (pid: string, epoch = projectEpochRef.current) => {
    const requestId = ++shotLoadRequestRef.current
    const mutation = shotMutationRef.current
    const list = await shotApi.list(pid)
    if (
      !isLatestResourceResponse(requestId, shotLoadRequestRef.current, mutation, shotMutationRef.current) ||
      !isCurrentProject(pid, epoch)
    ) return
    replaceShots(list || [])
  }

  const loadAssetBoard = async (pid: string, epoch = projectEpochRef.current) => {
    const requestId = ++assetLoadRequestRef.current
    const mutation = assetMutationRef.current
    const board = await assetApi.board(pid)
    if (
      !isLatestResourceResponse(requestId, assetLoadRequestRef.current, mutation, assetMutationRef.current) ||
      !isCurrentProject(pid, epoch)
    ) return null
    setAssetBoard({ characters: board.characters || [], scenes: board.scenes || [] })
    return board
  }

  const refreshWorkspaceData = async (pid: string, includeAssets = false, epoch = projectEpochRef.current) => {
    await loadProjectShots(pid, epoch)
    if (includeAssets) {
      await loadAssetBoard(pid, epoch)
    }
  }

  const loadStyleTemplates = async () => {
    const requestId = ++styleTemplateRequestRef.current
    const result = await settingsApi.styleTemplates()
    if (mountedRef.current && requestId === styleTemplateRequestRef.current && Array.isArray(result.templates)) {
      setStyleTemplates(result.templates)
    }
  }

  const applyProjectDetail = (projectDetail: any) => {
    setProject({
      projectId: projectDetail.id,
      parentProjectId: projectDetail.parent_project_id || '',
      parentProjectTitle: projectDetail.parent_project_title || '',
      projectType: projectDetail.project_type || 'series',
      episodeNumber: projectDetail.episode_number || 0,
      title: projectDetail.title,
      genre: projectDetail.genre,
      style: projectDetail.style || style,
      status: projectDetail.status,
      outputFormat: projectDetail.output_format || outputFormat,
      resolution: projectDetail.resolution || resolution,
      platform: projectDetail.platform || platform,
    })
    setEpisodeTitleDraft(projectDetail.title || '')
    if (projectDetail.effective_style || projectDetail.requested_style) {
      setEffectiveStyleMeta({
        requested_style: projectDetail.requested_style,
        effective_style: projectDetail.effective_style,
        style_source: projectDetail.style_source,
      })
    }
  }

  const activateProjectDetail = (projectDetail: any) => {
    const nextProjectId = String(projectDetail.id)
    const nextEpoch = projectEpochRef.current + 1
    projectEpochRef.current = nextEpoch
    activeProjectIdRef.current = nextProjectId
    activatedProjectRef.current = { projectId: nextProjectId, projectEpoch: nextEpoch }
    applyProjectDetail(projectDetail)
    return { projectId: nextProjectId, projectEpoch: nextEpoch }
  }

  const closeCurrentWebSocket = () => {
    wsClosedByUserRef.current = true
    wsEpochRef.current += 1
    if (wsReconnectTimerRef.current != null) {
      window.clearTimeout(wsReconnectTimerRef.current)
      wsReconnectTimerRef.current = null
    }
    wsRef.current?.close()
    wsRef.current = null
    wsProjectIdRef.current = null
    wsProjectEpochRef.current = 0
  }

  const connectWebSocket = (pid: string, projectEpoch = projectEpochRef.current) => {
    if (
      wsRef.current &&
      wsProjectIdRef.current === pid &&
      wsProjectEpochRef.current === projectEpoch &&
      (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return wsRef.current
    }

    if (wsReconnectTimerRef.current != null) {
      window.clearTimeout(wsReconnectTimerRef.current)
      wsReconnectTimerRef.current = null
    }
    // Invalidate callbacks from the previous socket before closing it. The
    // close event is asynchronous and must not schedule a reconnect for an
    // obsolete project.
    wsEpochRef.current += 1
    wsClosedByUserRef.current = true
    wsRef.current?.close()
    const socketEpoch = wsEpochRef.current + 1
    wsEpochRef.current = socketEpoch
    wsClosedByUserRef.current = false
    wsProjectIdRef.current = pid
    wsProjectEpochRef.current = projectEpoch

    const scheduleReconnect = () => {
      if (
        wsClosedByUserRef.current ||
        socketEpoch !== wsEpochRef.current ||
        !isCurrentProject(pid, projectEpoch) ||
        wsReconnectTimerRef.current != null
      ) {
        return
      }
      const attempt = wsReconnectAttemptRef.current
      const delay = Math.min(1000 * 2 ** Math.min(attempt, 4), 10000)
      wsReconnectAttemptRef.current = attempt + 1
      wsReconnectTimerRef.current = window.setTimeout(() => {
        wsReconnectTimerRef.current = null
        if (socketEpoch !== wsEpochRef.current || !isCurrentProject(pid, projectEpoch)) return
        connectWebSocket(pid, projectEpoch)
      }, delay)
    }

    // 单一 WebSocket 入口：统一驱动流程进度、镜头更新和完成态。
    const ws = createWebSocket(pid, (data) => {
      // A delayed frame from a previous project/socket must never mutate the
      // currently selected project.
      if (socketEpoch !== wsEpochRef.current || !isCurrentProject(pid, projectEpoch)) return
      const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false })
      // 实时读取最新运行模式：全自动模式下后端 LangGraph 不在人工卡点停留，
      // 前端也不应切到“等待确认”态或关闭生成中状态。
      const autoMode = useProjectStore.getState().runMode === 'auto'

      if (data.type === 'progress') {
        appendLog(`[${ts}] ${getStepLabel(data.step)} | ${data.progress ?? 0}%`)
        setProgress(Number(data.progress || 0), data.step || '')

        if (data.step === 'wait_asset_confirm') {
          if (autoMode) {
            void loadProjectShots(pid)
            void loadAssetBoard(pid)
            return
          }
          setGenerating(false)
          setLoading(false)
          setAssetBoardReady(true)
          setAwaitingStoryboardConfirm(false)
          setWorkspaceTab('assets')
          void loadProjectShots(pid)
          void loadAssetBoard(pid)
          message.info('角色板和场景板已生成，请确认素材后生成故事板')
          return
        }

        if (data.step === 'wait_storyboard_approval') {
          if (autoMode) {
            void loadProjectShots(pid)
            return
          }
          setGenerating(false)
          setLoading(false)
          setGeneratingStoryboard(false)
          setAwaitingStoryboardConfirm(true)
          setWorkspaceTab('review')
          void loadProjectShots(pid)
          message.info('故事板已生成，请逐镜头审核并生成视频')
          return
        }

        if (
          data.step === 'phase2_start' ||
          data.step === 'generate_voice' ||
          data.step === 'generate_seedance_video' ||
          data.step === 'compose_video' ||
          data.step === 'quality_check'
        ) {
          setAwaitingStoryboardConfirm(false)
          setGenerating(true)
        }
        return
      }

      if (data.type === 'shot_update' && data.shot_id) {
        appendLog(`[${ts}] 镜头已更新 | ${data.shot_id}`)
        // 字段级守卫：空媒体路径不覆盖已有有效路径；旧任务（版本号回退）
        // 的迟到响应整体丢弃，防止覆盖新生成的素材。
        applyServerShotUpdate(data.shot_id, data)
        // 只有真正落到 store 里的新视频才切换预览（被守卫丢弃的旧任务不触发）。
        const updated = useShotStore.getState().shots.find((item) => item.id === data.shot_id)
        if (updated?.video_path) {
          if (autoMode) {
            void loadProjectShots(pid)
          } else {
            setGenerating(false)
            setPreviewMode('video')
            setWorkspaceTab('video')
            void loadProjectShots(pid)
          }
        }
        return
      }

      if (data.type === 'complete') {
        appendLog(`[${ts}] 流程执行完成`)
        applyServerProjectTitle(pid, data.title)
        if (data.effective_style || data.requested_style) {
          setEffectiveStyleMeta({
            requested_style: data.requested_style,
            effective_style: data.effective_style,
            style_source: data.style_source,
          })
        }

        if (data.asset_board_ready) {
          if (autoMode) {
            if (Array.isArray(data.shots) && data.shots.length > 0) {
              replaceShots(data.shots)
            }
            void loadAssetBoard(pid)
            return
          }
          setGenerating(false)
          setLoading(false)
          setAssetBoardReady(true)
          setAwaitingStoryboardConfirm(false)
          setWorkspaceTab('assets')
          if (Array.isArray(data.shots) && data.shots.length > 0) {
            replaceShots(data.shots)
          }
          void loadAssetBoard(pid)
          return
        }

        if (!autoMode && (!data.video_path || awaitingRef.current || stepRef.current === 'wait_storyboard_approval')) {
          setGenerating(false)
          setLoading(false)
          setAwaitingStoryboardConfirm(true)
          setPreviewMode('shot')
          setWorkspaceTab('review')
          if (Array.isArray(data.shots) && data.shots.length > 0) {
            replaceShots(data.shots)
          }
          void loadProjectShots(pid)
          return
        }

        if (autoMode && !data.video_path) {
          // 全自动模式下的中间态 complete（无成片）：仅刷新数据，保持生成中，等待后续节点。
          if (Array.isArray(data.shots) && data.shots.length > 0) {
            replaceShots(data.shots)
          } else {
            void loadProjectShots(pid)
          }
          return
        }

        setGenerating(false)
        setLoading(false)
        setAwaitingStoryboardConfirm(false)
        setProgress(100, 'quality_check')

        if (data.video_path) {
          setVideoPath(data.video_path)
          setPreviewMode('video')
          setWorkspaceTab('video')
        }

        if (Array.isArray(data.shots) && data.shots.length > 0) {
          replaceShots(data.shots)
        } else {
          void loadProjectShots(pid)
        }

        message.success('生成完成')
        return
      }

      if (data.type === 'storyboard_ready') {
        appendLog(`[${ts}] 故事板生成完成${autoMode ? '（全自动继续生成视频）' : '，等待审核'}`)
        if (data.effective_style || data.requested_style) {
          setEffectiveStyleMeta({
            requested_style: data.requested_style,
            effective_style: data.effective_style,
            style_source: data.style_source,
          })
        }
        if (autoMode) {
          void loadProjectShots(pid)
          return
        }
        setGenerating(false)
        setGeneratingStoryboard(false)
        setAwaitingStoryboardConfirm(true)
        setAssetBoardReady(false)
        setWorkspaceTab('review')
        void loadProjectShots(pid)
        return
      }

      if (data.type === 'render_complete' && data.video_url) {
        appendLog(`[${ts}] 成片导出完成`)
        setVideoPath(data.video_url)
        setPreviewMode('video')
        setWorkspaceTab('video')
        setGenerating(false)
        setLoading(false)
        setProgress(100, 'quality_check')
        message.success('成片已生成，可直接播放')
        return
      }

      if (data.type === 'quality_review' && data.review) {
        // 自动模式质量门禁每轮审核的实时回显：只记日志 + 通知面板刷新，
        // 不在这里改 store（审核详情由面板按镜头拉取，避免部分更新）。
        const review = data.review as {
          shot_id: string
          verdict: string
          overall_score: number
          degraded?: boolean
          stage?: string
        }
        const meta = verdictMeta(review.verdict)
        appendLog(
          `[${ts}] 质量审核 | 镜头 ${review.shot_id} ${review.stage === 'video' ? '视频' : '故事板'} ${meta.label}` +
            `（${Math.round(Number(review.overall_score || 0) * 100)} 分）${review.degraded ? '｜含未检测维度（降级放行）' : ''}`,
        )
        window.dispatchEvent(new CustomEvent('quality-review-updated', { detail: { shot_id: review.shot_id } }))
        return
      }

      if (data.type === 'quality_gate_needs_human' && Array.isArray(data.shot_ids)) {
        appendLog(`[${ts}] 质量门禁 | ${data.shot_ids.length} 个镜头始终未通过自动审核，已转人工（needs_review）`)
        message.warning('部分镜头未通过自动质量审核，已转人工处理，请在「分镜审核」中查看评分与问题')
        return
      }

      if (data.type === 'error') {
        appendLog(`[${ts}] 错误 | ${data.message || '未知错误'}`)
        message.error(data.message || '流程执行失败')
        setGenerating(false)
        setLoading(false)
        setAwaitingStoryboardConfirm(false)
        void loadProjectShots(pid)
      }
    }, {
      onOpen: () => {
        if (socketEpoch === wsEpochRef.current) wsReconnectAttemptRef.current = 0
      },
      onClose: scheduleReconnect,
    })

    wsRef.current = ws
    return ws
  }

  const ensureProject = async (): Promise<{ projectId: string; projectEpoch: number }> => {
    const currentProjectId = useProjectStore.getState().projectId
    if (currentProjectId && activeProjectIdRef.current === currentProjectId) {
      return { projectId: currentProjectId, projectEpoch: projectEpochRef.current }
    }

    if (projectCreationPromiseRef.current) return projectCreationPromiseRef.current

    const navigationIntent = currentProjectNavigationIntent()
    initializingProjectRef.current = true
    initialScriptRef.current = script
    const operation = (async () => {
      try {
        const project = await projectApi.create({
          title: script.trim() ? script.slice(0, 20) : '未命名项目',
          first_episode_title: '第 1 集',
          style,
          genre: '',
        })
        if (
          !mountedRef.current ||
          useProjectStore.getState().projectId !== null ||
          currentProjectNavigationIntent() !== navigationIntent
        ) {
          throw new Error('项目上下文已变化，已忽略新建结果')
        }
        const activeProject = project.first_episode || project
        initializingProjectIdRef.current = activeProject.id as string
        return activateProjectDetail(activeProject)
      } catch (error) {
        initializingProjectRef.current = false
        initializingProjectIdRef.current = null
        initialScriptRef.current = ''
        throw error
      } finally {
        projectCreationPromiseRef.current = null
      }
    })()
    projectCreationPromiseRef.current = operation
    return operation
  }

  /** 剧本解析完成后，后端已根据剧本自动命名：同步标题输入框并刷新侧边栏项目列表。 */
  const applyServerProjectTitle = (pid: string, nextTitle: unknown) => {
    const title = typeof nextTitle === 'string' ? nextTitle.trim() : ''
    if (!title) return
    if (useProjectStore.getState().projectId === pid && useProjectStore.getState().title !== title) {
      setProject({ title })
    }
    window.dispatchEvent(new CustomEvent(PROJECTS_REFRESHED_EVENT))
  }

  const updateProjectField = async (field: 'style' | 'resolution' | 'title', value: string) => {
    const previousValue = ({ style, resolution, title } as Record<string, string>)[field]
    setProject({ [field]: value } as any)
    if (!projectId) return
    const operation = beginOperation(`project-field:${field}`, projectId)
    if (!isCurrentOperation(operation)) return
    try {
      const result = await projectApi.update(projectId, { [field]: value })
      if (field === 'style') {
        setEffectiveStyleMeta({
          requested_style: result.requested_style || value,
          effective_style: result.effective_style || value,
          style_source: result.style_source || 'project_request',
        })
        if (result.assets_stale) {
          message.warning('画风已切换：旧素材已保留并标记待重新生成，建议先重建资产，再批量重生成受影响镜头')
        } else {
          message.warning('画风已切换：旧分镜素材已保留并标记待重新生成，请在分镜中选择镜头重新生成')
        }
      }
      if (field === 'style' || field === 'resolution') {
        // 生成配置变化会把全部镜头标记待重生成（旧素材保留）：
        // 刷新列表，让缩略图与预览立即显示「待重生成」状态。
        void loadProjectShots(projectId)
      }
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      setProject({ [field]: previousValue } as any)
      message.error('项目配置更新失败：' + (err.message || '未知错误'))
    }
  }

  const handleRebuildAssets = async () => {
    if (!projectId || rebuildingAssets) return
    setRebuildingAssets(true)
    try {
      await projectApi.rebuildAssets(projectId)
      message.success('已开始按当前画风重建角色板与场景板（旧文件保留，仅解除引用）')
      void loadAssetBoard(projectId)
    } catch (err: any) {
      message.error('重建资产失败：' + (err.message || '未知错误'))
    } finally {
      setRebuildingAssets(false)
    }
  }

  const startEditAsset = (item: any) => {
    setEditingAssetId(item.id)
    setAssetDraft({
      ...item,
      appearanceText: item.appearance?.description || item.appearance?.default_outfit || '',
      keyFeaturesText: Array.isArray(item.key_features) ? item.key_features.join('\n') : '',
    })
  }

  const updateAssetDraft = (key: string, value: any) => {
    setAssetDraft((draft) => ({ ...draft, [key]: value }))
  }

  const handleSaveAsset = async (regenerate = false) => {
    if (!projectId || !editingAssetId) return
    const entryProjectId = projectId
    const assetId = editingAssetId
    const tab = assetTab
    const draft = { ...assetDraft }
    const operation = beginOperation('save-asset', entryProjectId)
    try {
      setSavingAsset(true)
      const keyFeatures = String(draft.keyFeaturesText || '')
        .split('\n')
        .map((item) => item.trim())
        .filter(Boolean)
      const payload =
        tab === 'characters'
          ? {
              project_id: entryProjectId,
              name: draft.name || '',
              appearance: { ...(draft.appearance || {}), description: draft.appearanceText || '' },
              personality: draft.personality || '',
              visual_prompt: draft.visual_prompt || '',
              negative_prompt: draft.negative_prompt || '',
              voice_id: draft.voice_id || '',
              key_features: keyFeatures,
              default_outfit: draft.default_outfit || '',
              lora_profile: draft.lora_profile || '',
              ip_adapter_profile: draft.ip_adapter_profile || '',
              wardrobe_lock: draft.wardrobe_lock || '',
              seed: String(draft.seed || '42'),
              regenerate,
            }
          : {
              project_id: entryProjectId,
              name: draft.name || '',
              description: draft.description || '',
              visual_prompt: draft.visual_prompt || '',
              negative_prompt: draft.negative_prompt || '',
              key_features: keyFeatures,
              scene_group_key: draft.scene_group_key || '',
              time_of_day: draft.time_of_day || '',
              prop_lock: draft.prop_lock || '',
              seed: Number(draft.seed || 1200),
              regenerate,
            }
      const updated =
        tab === 'characters'
          ? await assetApi.updateCharacter(assetId, payload)
          : await assetApi.updateScene(assetId, payload)
      if (!isCurrentOperation(operation)) return
      assetMutationRef.current += 1
      setAssetBoard((board) => {
        if (!board) return board
        const key = tab
        return {
          ...board,
          [key]: board[key].map((item: any) => (item.id === assetId ? updated : item)),
        }
      })
      setEditingAssetId(null)
      setAssetDraft({})
      message.success(regenerate ? '素材已更新并重生成' : '素材已保存')
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      message.error('素材保存失败：' + (err.message || '未知错误'))
    } finally {
      if (isCurrentOperation(operation)) setSavingAsset(false)
    }
  }

  const commitEpisodeTitle = async () => {
    const nextTitle = episodeTitleDraft.trim() || title || '第 1 集'
    setEpisodeTitleDraft(nextTitle)
    if (nextTitle !== title) {
      await updateProjectField('title', nextTitle)
    }
  }

  const handleCreateProject = async () => {
    beginProjectNavigationIntent()
    const sourceProjectId = useProjectStore.getState().projectId
    const operation = beginOperation('create-project', sourceProjectId)
    try {
      setCreatingProject(true)
      const canNavigate = await requestProjectNavigation(sourceProjectId, null)
      if (!canNavigate || !isCurrentOperation(operation)) return
      const nextTitle = newProjectTitle.trim() || '未命名项目'
      const project = await projectApi.create({
        title: nextTitle,
        first_episode_title: newEpisodeTitle.trim() || '第 1 集',
        style,
        genre: '',
        output_format: outputFormat,
        resolution,
        platform,
      })
      if (!isCurrentOperation(operation)) return
      const activeProject = project.first_episode || project
      activateProjectDetail(activeProject)

      setShots([])
      selectShot(null)
      setScript('')
      setGenerating(false)
      setAwaitingStoryboardConfirm(false)
      setAssetBoardReady(false)
      setAssetBoard(null)
      setVideoPath('')
      setProgress(0, '')
      setWorkspaceTab('script')
      clearLogs()
      setNewProjectTitle('')
      setNewEpisodeTitle('第 1 集')
      setShowCreatePanel(false)
      message.success('新建项目成功，已自动创建第一集')
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      message.error('新建项目失败：' + (err.message || '未知错误'))
    } finally {
      if (isLatestOperation(operation) && mountedRef.current) setCreatingProject(false)
    }
  }

  /**
   * 付费任务入口的统一估算闸门：返回 true 才真正提交。
   *
   * 项目还不存在时（首次创建项目并解析剧本）无法估算——项目要到提交时才创建，
   * 这时跳过确认，由后端的硬预算拦截兜底。
   */
  const confirmTaskEstimate = async (
    jobType: 'script_pipeline' | 'storyboard' | 'asset_generation' | 'shot_image' | 'shot_audio' | 'shot_video' | 'render',
    entryLabel: string,
    options: { shotId?: string; shotIds?: string[] } = {},
  ): Promise<boolean> => {
    if (!projectId) return true
    return estimateGate.confirm({
      job_type: jobType,
      project_id: projectId,
      shot_id: options.shotId,
      shot_ids: options.shotIds,
      entryLabel,
    })
  }

  const submitScriptForStoryboard = async (nextScript: string, existingOperation?: ProjectOperation) => {
    if (!nextScript.trim()) {
      message.warning('请输入剧本内容')
      return
    }
    if (!(await confirmTaskEstimate('script_pipeline', '剧本解析'))) return
    let operation = existingOperation || beginOperation('pipeline')

    setLoading(true)
    setGenerating(true)
    setAwaitingStoryboardConfirm(false)
    setAssetBoardReady(false)
    setVideoPath('')
    setPreviewMode('shot')
    setWorkspaceTab('script')
    clearLogs()
    setProgress(0, 'parse_script')

    try {
      const context = await ensureProject()
      const boundOperation = bindOperationToProject(operation, context)
      if (!boundOperation) return
      operation = boundOperation
      connectWebSocket(context.projectId, context.projectEpoch)
      const parseResult = await scriptApi.parse({
        project_id: context.projectId,
        user_input: nextScript,
        input_type: 'text',
        style,
        output_format: outputFormat,
        resolution,
        platform,
        target_duration: 30,
        mode: runMode,
      })
      if (!isCurrentOperation(operation)) return
      notifyBudgetWarning(parseResult)
      appendLog(
        `[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] 已提交解析任务（${runMode === 'auto' ? '全自动生成' : '手动审核'}）`,
      )
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      if (!notifyBudgetBlocked(err) && !notifyProviderBlocked(err)) message.error('提交失败：' + (err.message || '未知错误'))
      setGenerating(false)
      setLoading(false)
    }
  }

  const handleGenerate = async () => {
    await submitScriptForStoryboard(script)
  }

  const handleAutoWriteScript = async () => {
    let operation = beginOperation('pipeline')
    try {
      setAutoWriting(true)
      setGenerating(true)
      setAwaitingStoryboardConfirm(false)
      setAssetBoardReady(false)
      setVideoPath('')
      clearLogs()
      setWorkspaceTab('script')
      setProgress(0, 'generate_script')

      const context = await ensureProject()
      const boundOperation = bindOperationToProject(operation, context)
      if (!boundOperation) return
      operation = boundOperation
      connectWebSocket(context.projectId, context.projectEpoch)
      const prompt = script.trim() || '一个适合 45 秒竖屏漫剧的温暖成长故事'
      const result = await scriptApi.generate({
        project_id: context.projectId,
        prompt,
        style,
        genre: '原创短剧',
        target_duration: 45,
      })
      if (!isCurrentOperation(operation)) return
      const generatedScript = result.script || ''
      setScript(generatedScript)
      appendLog(`[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] Agent 已生成完整剧本`)
      await submitScriptForStoryboard(generatedScript, operation)
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      message.error('自动生成剧本失败：' + (err.message || '未知错误'))
      setGenerating(false)
      setLoading(false)
    } finally {
      if (isLatestOperation(operation) && mountedRef.current) setAutoWriting(false)
    }
  }

  const handleUpload = async (file: File) => {
    if (!(await confirmTaskEstimate('script_pipeline', '剧本上传解析'))) return
    let operation = beginOperation('pipeline')
    try {
      setUploading(true)
      setGenerating(true)
      setAwaitingStoryboardConfirm(false)
      setAssetBoardReady(false)
      setVideoPath('')
      setPreviewMode('shot')
      setWorkspaceTab('script')
      clearLogs()
      setProgress(0, 'parse_script')

      const context = await ensureProject()
      const boundOperation = bindOperationToProject(operation, context)
      if (!boundOperation) return
      operation = boundOperation
      connectWebSocket(context.projectId, context.projectEpoch)

      const formData = new FormData()
      formData.append('project_id', context.projectId)
      formData.append('file', file)
      formData.append('style', style)
      formData.append('output_format', outputFormat)
      formData.append('resolution', resolution)
      formData.append('platform', platform)
      formData.append('mode', runMode)

      const result = await scriptApi.upload(formData)
      if (!isCurrentOperation(operation)) return
      if (typeof result.script === 'string') {
        setScript(result.script)
      }
      notifyBudgetWarning(result)
      appendLog(`[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] 已上传剧本：${file.name}`)
      message.success('剧本上传成功')
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      if (!notifyBudgetBlocked(err) && !notifyProviderBlocked(err)) message.error('上传失败：' + (err.message || '未知错误'))
      setGenerating(false)
    } finally {
      if (isLatestOperation(operation) && mountedRef.current) setUploading(false)
    }
  }

  const handleGenerateSelectedShotVideo = async () => {
    if (!projectId || !selectedShot) return
    const entryProjectId = projectId
    const shot = selectedShot
    const operation = beginOperation(`generate-video:${shot.id}`, entryProjectId)

    try {
      setConfirming(true)
      if (!shot.confirmed) {
        message.warning('请先审核通过当前镜头故事板')
        return
      }
      if (!(shot.storyboard_path || shot.image_path)) {
        message.warning('当前镜头尚未生成定稿故事板')
        return
      }
      if (!(await confirmTaskEstimate('shot_video', '生成镜头 ' + shot.sequence + ' 视频', { shotId: shot.id }))) return
      setGenerating(true)
      setPreviewMode('shot')
      setWorkspaceTab('video')
      connectWebSocket(entryProjectId, operation.projectEpoch)
      const videoResult = await shotApi.generateVideo(shot.id, Boolean(shot.video_path))
      if (!isCurrentOperation(operation)) return
      setAwaitingStoryboardConfirm(false)
      notifyBudgetWarning(videoResult)
      appendLog(`[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] 已提交镜头 ${shot.sequence} 视频生成`)
      message.success('当前镜头视频生成已启动')
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      if (!notifyBudgetBlocked(err) && !notifyProviderBlocked(err)) message.error('镜头视频生成失败：' + (err.message || '未知错误'))
      setGenerating(false)
    } finally {
      if (isLatestOperation(operation) && mountedRef.current) setConfirming(false)
    }
  }

  const handleComposeProjectVideo = async () => {
    if (!projectId) return
    if (!allShotVideosReady) {
      message.warning('请先完成每个镜头的视频生成')
      return
    }
    if (!(await confirmTaskEstimate('render', '导出成片'))) return
    const entryProjectId = projectId
    const operation = beginOperation('compose-video', entryProjectId)

    try {
      setComposing(true)
      setGenerating(true)
      setPreviewMode('video')
      setWorkspaceTab('video')
      setProgress(92, 'compose_video')
      connectWebSocket(entryProjectId, operation.projectEpoch)
      const renderResult = await renderApi.start({ project_id: entryProjectId, output_format: outputFormat, resolution })
      if (!isCurrentOperation(operation)) return
      notifyBudgetWarning(renderResult)
      appendLog(`[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] 已提交成片合成任务`)
      message.success('成片合成已启动')
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      if (!notifyBudgetBlocked(err) && !notifyProviderBlocked(err)) message.error('成片合成失败：' + (err.message || '未知错误'))
      setGenerating(false)
    } finally {
      if (isLatestOperation(operation) && mountedRef.current) setComposing(false)
    }
  }

  const handleApproveShot = async (shotId: string, approved: boolean) => {
    if (!projectId) return
    const operation = beginOperation(`approve-shot:${shotId}`, projectId)
    try {
      await shotApi.approveStoryboard(shotId, approved)
      if (!isCurrentOperation(operation)) return
      updateShot(shotId, { confirmed: approved, status: approved ? 'storyboard_approved' : 'needs_review' })
      message.success(approved ? '该镜头已通过审核' : '已标记为需调整')
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      message.error('审核操作失败：' + (err.message || '未知错误'))
    }
  }

  const changePreviewScale = (delta: number) => {
    setPreviewScale((value) => Math.min(3, Math.max(0.5, Number((value + delta).toFixed(2)))))
  }

  const resetPreviewTransform = () => {
    setPreviewScale(1)
    setPreviewOffset({ x: 0, y: 0 })
    setDragStart(null)
  }

  const beginPreviewDrag = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!imageUrl || isGenerating || previewMode !== 'shot') return
    event.preventDefault()
    setDragStart({ x: event.clientX, y: event.clientY, ox: previewOffset.x, oy: previewOffset.y })
  }

  const movePreviewDrag = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart) return
    setPreviewOffset({
      x: dragStart.ox + event.clientX - dragStart.x,
      y: dragStart.oy + event.clientY - dragStart.y,
    })
  }

  const handleGenerateStoryboard = async () => {
    if (!projectId) return
    if (!(await confirmTaskEstimate('storyboard', '生成故事板'))) return
    const entryProjectId = projectId
    const operation = beginOperation('generate-storyboard', entryProjectId)

    try {
      setGeneratingStoryboard(true)
      setGenerating(true)
      setAwaitingStoryboardConfirm(false)
      setWorkspaceTab('storyboard')
      connectWebSocket(entryProjectId, operation.projectEpoch)
      const storyboardResult = await shotApi.generateStoryboard(entryProjectId)
      if (!isCurrentOperation(operation)) return
      notifyBudgetWarning(storyboardResult)
      appendLog(`[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] 已确认素材，开始生成定稿故事板参考图`)
      message.success('故事板任务已启动')
    } catch (err: any) {
      if (!isCurrentOperation(operation)) return
      if (!notifyBudgetBlocked(err) && !notifyProviderBlocked(err)) message.error('故事板生成失败：' + (err.message || '未知错误'))
      setGenerating(false)
    } finally {
      if (isLatestOperation(operation) && mountedRef.current) setGeneratingStoryboard(false)
    }
  }

  useEffect(() => {
    generateRef.current = handleGenerate
  }, [script, projectId, style, outputFormat, resolution, platform, runMode])

  useEffect(() => {
    awaitingRef.current = awaitingStoryboardConfirm
    stepRef.current = currentStep
  }, [awaitingStoryboardConfirm, currentStep])

  useEffect(() => {
    setEpisodeTitleDraft(title)
  }, [title])

  useEffect(() => {
    const onParse = () => {
      void generateRef.current()
    }

    const onOpenCreateProject = () => {
      setWorkspaceTab('script')
      setShowCreatePanel(true)
    }

    window.addEventListener(PARSE_SCRIPT_EVENT, onParse)
    window.addEventListener(OPEN_CREATE_PROJECT_EVENT, onOpenCreateProject)
    return () => {
      window.removeEventListener(PARSE_SCRIPT_EVENT, onParse)
      window.removeEventListener(OPEN_CREATE_PROJECT_EVENT, onOpenCreateProject)
    }
  }, [])

  useEffect(() => {
    const reloadStyleTemplates = () => {
      void loadStyleTemplates().catch(() => undefined)
    }

    reloadStyleTemplates()
    window.addEventListener(STYLE_TEMPLATES_UPDATED_EVENT, reloadStyleTemplates)
    return () => window.removeEventListener(STYLE_TEMPLATES_UPDATED_EVENT, reloadStyleTemplates)
  }, [])

  useEffect(() => {
    const activated = projectId && activatedProjectRef.current?.projectId === projectId
      ? activatedProjectRef.current
      : null
    const epoch = activated?.projectEpoch ?? projectEpochRef.current + 1
    projectEpochRef.current = epoch
    activatedProjectRef.current = null
    activeProjectIdRef.current = projectId
    shotLoadRequestRef.current += 1
    assetLoadRequestRef.current += 1
    assetMutationRef.current += 1
    // React StrictMode mounts effects twice in development. Tie the flag to
    // the created project ID so both setup passes preserve the submitted
    // script, while a later project switch still performs a full reset.
    const preserveInitialRun =
      initializingProjectRef.current && initializingProjectIdRef.current === projectId
    const preservedScript = preserveInitialRun ? initialScriptRef.current : ''
    if (!preserveInitialRun) {
      initializingProjectRef.current = false
      initializingProjectIdRef.current = null
      initialScriptRef.current = ''
    }

    // Clear all project-scoped state before loading the next project. This is
    // intentionally synchronous so an old project's script/media cannot be
    // displayed while the new project requests are in flight.
    setShots([])
    selectShot(null)
    setScript(preservedScript)
    setAssetBoard(null)
    setAssetBoardReady(false)
    setEditingAssetId(null)
    setAssetDraft({})
    setSavingAsset(false)
    setVideoPath('')
    setAwaitingStoryboardConfirm(false)
    setLoading(false)
    setAutoWriting(false)
    setUploading(false)
    setConfirming(false)
    setComposing(false)
    setGeneratingStoryboard(false)
    setImagePreview(null)
    if (!preserveInitialRun) {
      setGenerating(false)
      setProgress(0, '')
      clearLogs()
    }
    setWorkspaceTab('script')
    setPreviewMode('shot')
    resetPreviewTransform()

    if (!projectId) {
      closeCurrentWebSocket()
      return () => {
        if (projectEpochRef.current === epoch) activeProjectIdRef.current = null
      }
    }

    connectWebSocket(projectId, epoch)
    projectApi.get(projectId)
      .then((projectDetail) => {
        if (!isCurrentProject(projectId, epoch)) return
        applyProjectDetail(projectDetail)
        if (!preserveInitialRun || projectDetail.input_text) {
          setScript(projectDetail.input_text || '')
        }
        setVideoPath(projectDetail.video_path || '')
      })
      .catch(() => undefined)
    void loadProjectShots(projectId, epoch).catch(() => undefined)
    void loadAssetBoard(projectId, epoch).catch(() => undefined)
    return () => {
      if (projectEpochRef.current !== epoch) return
      activeProjectIdRef.current = null
      closeCurrentWebSocket()
    }
  }, [projectId])

  useEffect(() => {
    if (!projectId || !isGenerating) return

    const epoch = projectEpochRef.current
    const timer = window.setInterval(() => {
      if (!isCurrentProject(projectId, epoch)) return
      if (pollingRef.current) return
      pollingRef.current = true
      const includeAssets = currentStep === 'wait_asset_confirm' || currentStep === 'generate_storyboard'
      refreshWorkspaceData(projectId, includeAssets, epoch)
        .catch(() => undefined)
        .finally(() => {
          pollingRef.current = false
        })
    }, 3200)

    return () => window.clearInterval(timer)
  }, [projectId, isGenerating, currentStep])

  const imageUrl = toOutputUrl(selectedShot?.storyboard_path || selectedShot?.image_path)
  const videoUrl = toOutputUrl(videoPath)
  const currentVideoUrl = selectedShotVideoUrl || videoUrl

  useEffect(() => {
    resetPreviewTransform()
  }, [selectedShot?.id, imageUrl])

  const assetItems = assetTab === 'characters' ? assetBoard?.characters || [] : assetBoard?.scenes || []

  const renderReferencePreview = (item: any, label: string) => {
    const firstImage = Array.isArray(item.reference_images) ? item.reference_images[0] : ''
    const previewUrl = toOutputUrl(firstImage)
    return previewUrl ? (
      <button
        type="button"
        className="image-preview-trigger"
        onClick={() => setImagePreview({ url: previewUrl, title: item.name || label })}
        aria-label="查看高清原图"
      >
        <img src={previewUrl} alt={label} loading="lazy" decoding="async" />
      </button>
    ) : (
      <span>{label}</span>
    )
  }

  const selectedShotReady = Boolean(selectedShot && (selectedShot.storyboard_path || selectedShot.image_path))
  const showPreviewSurface = workspaceTab === 'storyboard' || workspaceTab === 'review' || workspaceTab === 'video'

  const toggleQueueShot = (shotId: string) => {
    setQueueSelectedIds((current) => current.includes(shotId)
      ? current.filter((id) => id !== shotId)
      : [...current, shotId])
  }

  const submitSelectiveRegeneration = async () => {
    if (!projectId || !queueSelectedIds.length || !queueStages.length) {
      message.warning('请选择镜头和至少一个重生成阶段')
      return
    }
    const locked = shots.filter((shot) => queueSelectedIds.includes(shot.id) && shot.confirmed)
    if (locked.length && !queueForceConfirmed) {
      message.warning(`已跳过 ${locked.length} 个已确认镜头；如需处理请开启“强制确认”`)
    }
    if (locked.length && queueForceConfirmed) {
      const confirmed = await new Promise<boolean>((resolve) => {
        Modal.confirm({
          title: '确认重新生成已锁定镜头？',
          content: `将处理 ${locked.length} 个已确认镜头，并使新的故事板回到待审核状态。`,
          okText: '继续重生成',
          cancelText: '取消',
          onOk: () => resolve(true),
          onCancel: () => resolve(false),
        })
      })
      if (!confirmed) return
    }
    setQueueSubmitting(true)
    try {
      const response = await regenerationQueueApi.submit({
        project_id: projectId,
        shot_ids: queueSelectedIds,
        stages: queueStages,
        priority: queuePriority,
        concurrency: queueConcurrency,
        order: queueOrder,
        reuse_audio: queueReuseAudio,
        resume_missing: queueResumeMissing,
        force_confirmed: queueForceConfirmed,
        version_map: queueVersion > 0
          ? Object.fromEntries(queueSelectedIds.map((id) => [id, queueVersion]))
          : {},
      })
      const blockedCount = Array.isArray(response?.blocked) ? response.blocked.length : 0
      message.success(`已加入队列${blockedCount ? `，${blockedCount} 项被阻塞` : ''}`)
      setQueueSelectedIds([])
      useTaskStore.getState().sync({ silent: true }).catch(() => undefined)
    } catch (error: any) {
      message.error(error?.response?.data?.detail || error?.message || '队列提交失败')
    } finally {
      setQueueSubmitting(false)
    }
  }

  // 切换工作区标签：同时同步预览模式（点击与键盘操作共用同一条路径）
  const selectWorkspaceTab = (tabId: WorkspaceTab) => {
    setWorkspaceTab(tabId)
    if (tabId === 'video' && currentVideoUrl) {
      setPreviewMode('video')
    }
    if (tabId === 'storyboard' || tabId === 'review') {
      setPreviewMode('shot')
    }
  }

  // ARIA 标签页「自动激活」模式：方向键 / Home / End 同时移动焦点与选中项，
  // 其余按键不拦截，保持回车、Tab、输入等默认行为。
  const handleWorkspaceTabKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    tabId: WorkspaceTab,
  ) => {
    const nextTabId = resolveWorkspaceTabFocus(
      event.key,
      WORKSPACE_TABS.map((tab) => tab.id),
      tabId,
    )
    if (!nextTabId) return
    event.preventDefault()
    selectWorkspaceTab(nextTabId)
    workspaceTabRefs.current.get(nextTabId)?.focus()
  }

  const openShotConfig = (shotId: string) => {
    selectShot(shotId)
    setWorkspaceTab('review')
    setPreviewMode('shot')
    window.dispatchEvent(new CustomEvent(WORKSPACE_NAVIGATE_EVENT, { detail: { tab: 'review', previewMode: 'shot' } }))
    window.dispatchEvent(new CustomEvent(OPEN_SHOT_CONFIG_EVENT))
  }

  return (
    <section className="main-workspace tabbed-workspace" aria-label="主工作区">
      <div className="workspace-tabbar" role="tablist" aria-label="工作区模块">
        {WORKSPACE_TABS.map((tab) => {
          const active = workspaceTab === tab.id
          const tabMeta =
            tab.id === 'review' && shots.length > 0
              ? `${approvedShotCount}/${shots.length}`
              : tab.id === 'assets' && (assetBoardReady || assetBoard)
                ? '已就绪'
                : tab.id === 'video' && currentVideoUrl
                  ? '可预览'
                  : ''

          return (
            <button
              key={tab.id}
              type="button"
              {...getWorkspaceTabAriaProps(tab.id, workspaceTab)}
              ref={(node) => {
                if (node) workspaceTabRefs.current.set(tab.id, node)
                else workspaceTabRefs.current.delete(tab.id)
              }}
              className={`workspace-tab${active ? ' active' : ''}`}
              onClick={() => selectWorkspaceTab(tab.id)}
              onKeyDown={(event) => handleWorkspaceTabKeyDown(event, tab.id)}
            >
              <span>{tab.label}</span>
              {tabMeta && <em>{tabMeta}</em>}
            </button>
          )
        })}
      </div>

      <div className="workspace-tab-body">
        {workspaceTab === 'script' && (
          <div
            className="script-panel panel-enter"
            {...getWorkspacePanelAriaProps('script')}
          >
        <div className="script-scope-bar">
          <div className="script-scope-copy">
            <span>{parentProjectTitle || (projectType === 'series' ? title : '未选择大项目')}</span>
            <strong>{projectType === 'episode' ? '当前剧集' : '当前项目'}</strong>
          </div>
          <Input
            size="small"
            value={episodeTitleDraft}
            className="episode-title-input"
            placeholder="请输入剧集名称"
            onChange={(event) => setEpisodeTitleDraft(event.target.value)}
            onBlur={() => void commitEpisodeTitle()}
            onPressEnter={() => void commitEpisodeTitle()}
          />
        </div>
        <BudgetSummaryPanel projectId={projectId} title={title} />
        <div className="workflow-rail" aria-label="创作流程">
          {['新建剧集', '上传剧本', 'AI解析', '资产板', '批量分镜', '逐镜审核', '生成视频'].map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
        <div className="workspace-toolbar">
          <div className="workspace-toolbar-controls">
            <div className="toolbar-field">
              <span className="toolbar-label">画风</span>
              <Select
                value={style}
                size="small"
                style={{ width: 160 }}
                onChange={(v) => void updateProjectField('style', v)}
                options={styleTemplates.map((item) => ({
                  value: item.value,
                  label: item.custom ? `${item.label}（自定义）` : item.label,
                }))}
              />
              <em className="toolbar-hint">
                {STYLE_DESCRIPTIONS[style] || styleTemplates.find((item) => item.value === style)?.keywords || STYLE_DESCRIPTIONS.anime}
              </em>
              {effectiveStyleMeta?.effective_style ? (
                <Tooltip
                  title={
                    effectiveStyleMeta.style_source === 'skill_override'
                      ? `Skill 方案覆盖了项目画风：请求 ${effectiveStyleMeta.requested_style}，实际生效 ${effectiveStyleMeta.effective_style}`
                      : `全链路（剧本/分镜/图像/视频）实际生效画风：${effectiveStyleMeta.effective_style}（来源：${
                          effectiveStyleMeta.style_source === 'skill_override' ? 'Skill 覆盖' : '项目/本次请求'
                        }）`
                  }
                >
                  <span className="toolbar-hint">
                    实际生效：
                    {styleTemplates.find((item) => item.value === effectiveStyleMeta.effective_style)?.label ||
                      effectiveStyleMeta.effective_style}
                    {effectiveStyleMeta.style_source === 'skill_override' ? '（Skill 覆盖）' : ''}
                    {effectiveStyleMeta.effective_style !== style ? '（与当前选择不一致，注意重建资产）' : ''}
                  </span>
                </Tooltip>
              ) : null}
              <Button
                size="small"
                type="default"
                loading={rebuildingAssets}
                onClick={() => void handleRebuildAssets()}
                disabled={!projectId}
                title="切换画风后，旧的角色三视图/场景基准图会标记失效；点击按当前画风重新生成（旧文件保留）"
              >
                重建资产
              </Button>
            </div>
            <div className="toolbar-field">
              <span className="toolbar-label">分辨率</span>
              <Select
                value={resolution}
                size="small"
                style={{ width: 130 }}
                onChange={(v) => void updateProjectField('resolution', v)}
                options={[
                  { value: '720p', label: '高清 720' },
                  { value: '1080p', label: '全高清 1080' },
                  { value: '2k', label: '影院级 2 千' },
                  { value: '4k', label: '超高清 4 千' },
                ]}
              />
            </div>
            <div className="toolbar-field">
              <span className="toolbar-label">生成模式</span>
              <Tooltip
                title={
                  runMode === 'auto'
                    ? '全自动：解析剧本后由 LangGraph 一路跑到成片，自动通过故事板与逐镜视频，无需人工审核。'
                    : '手动审核：每一步生成后暂停，等待你确认素材、逐镜审核故事板再生成视频。'
                }
              >
                <Segmented
                  size="small"
                  value={runMode}
                  onChange={(value) => setProject({ runMode: value as 'manual' | 'auto' })}
                  options={[
                    { value: 'manual', label: '手动审核' },
                    { value: 'auto', label: '全自动生成' },
                  ]}
                />
              </Tooltip>
            </div>
          </div>
        </div>

        {showCreatePanel && (
          <div className="workspace-create-panel">
            <div className="workspace-create-copy">
              <div className="workspace-create-title">新建项目</div>
              <div className="workspace-create-note">在主工作区完成项目创建，创建后会自动切换到新项目。</div>
            </div>
            <div className="workspace-create-form">
              <Input
                value={newProjectTitle}
                placeholder="请输入项目名称"
                onChange={(e) => setNewProjectTitle(e.target.value)}
                onPressEnter={() => void handleCreateProject()}
              />
              <Input
                value={newEpisodeTitle}
                placeholder="请输入第一集名称"
                onChange={(e) => setNewEpisodeTitle(e.target.value)}
                onPressEnter={() => void handleCreateProject()}
              />
              <Button type="primary" loading={creatingProject} onClick={() => void handleCreateProject()}>
                确认创建
              </Button>
              <Button onClick={() => setShowCreatePanel(false)}>取消</Button>
            </div>
          </div>
        )}

        <div className="script-editor-row">
          <TextArea
            value={script}
            onChange={(e) => setScript(e.target.value)}
            placeholder={'请输入剧本内容...\n\n场景：深夜街道，细雨\n人物：男主，情绪低沉\n镜头：中景，慢慢推近'}
            style={{
              height: '100%',
              resize: 'none',
              fontSize: 13,
              lineHeight: 1.7,
            }}
          />

          <div className="script-actions">
            <Button
              icon={<BulbOutlined />}
              onClick={() => void handleAutoWriteScript()}
              loading={autoWriting}
              size="small"
            >
              AI写剧本
            </Button>

            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={() => void handleGenerate()}
              loading={loading}
              size="small"
            >
              AI解析剧本
            </Button>

            <Button
              icon={<UploadOutlined />}
              size="small"
              loading={uploading}
              onClick={() => uploadInputRef.current?.click()}
            >
              上传剧本
            </Button>

            <input
              ref={uploadInputRef}
              type="file"
              accept=".txt,.docx"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) {
                  void handleUpload(file)
                }
                e.target.value = ''
              }}
            />
          </div>
        </div>
          </div>
        )}

        {workspaceTab === 'assets' && (
          <div
            className="asset-page panel-enter"
            {...getWorkspacePanelAriaProps('assets')}
          >
            <div className="asset-page-head">
              <div>
                <div className="asset-board-title">项目素材板</div>
                <div className="asset-board-note">
                  角色与场景将作为项目级素材复用到本集故事板和后续成片。
                </div>
              </div>
              <Button
                type="primary"
                size="small"
                loading={generatingStoryboard}
                disabled={!shots.length}
                onClick={() => void handleGenerateStoryboard()}
              >
                批量生成分镜
              </Button>
            </div>

            {(assetBoardReady || assetBoard) ? (
              <div className="asset-board-panel asset-page-board">
                <div className="asset-board-lists">
                  <div className="asset-tabs" aria-label="素材类型">
                    <button type="button" className={assetTab === 'characters' ? 'active' : ''} onClick={() => setAssetTab('characters')}>
                      角色板
                    </button>
                    <button type="button" className={assetTab === 'scenes' ? 'active' : ''} onClick={() => setAssetTab('scenes')}>
                      场景板
                    </button>
                  </div>
                  <div className="asset-card-strip asset-page-strip">
                    {assetItems.map((item) => (
                      <div className="asset-mini-card" key={item.id}>
                        <div className="asset-thumb">
                          {assetTab === 'characters'
                            ? renderReferencePreview(item, '三视图待生成')
                            : renderReferencePreview(item, '场景预览待生成')}
                        </div>
                        {editingAssetId === item.id ? (
                          <div className="asset-edit-form">
                            <Input size="small" value={assetDraft.name || ''} onChange={(event) => updateAssetDraft('name', event.target.value)} />
                            <TextArea
                              autoSize={{ minRows: 2, maxRows: 4 }}
                              value={assetTab === 'characters' ? assetDraft.personality || '' : assetDraft.description || ''}
                              onChange={(event) => updateAssetDraft(assetTab === 'characters' ? 'personality' : 'description', event.target.value)}
                            />
                            <TextArea
                              autoSize={{ minRows: 2, maxRows: 5 }}
                              value={assetDraft.visual_prompt || ''}
                              onChange={(event) => updateAssetDraft('visual_prompt', event.target.value)}
                              placeholder="视觉提示词"
                            />
                            <Input
                              size="small"
                              value={assetDraft.negative_prompt || ''}
                              onChange={(event) => updateAssetDraft('negative_prompt', event.target.value)}
                              placeholder="负向提示词"
                            />
                            <Input
                              size="small"
                              value={assetTab === 'characters' ? assetDraft.default_outfit || '' : assetDraft.prop_lock || ''}
                              onChange={(event) => updateAssetDraft(assetTab === 'characters' ? 'default_outfit' : 'prop_lock', event.target.value)}
                              placeholder={assetTab === 'characters' ? '服装锁定' : '道具/光源锁定'}
                            />
                            <div className="asset-edit-actions">
                              <Button size="small" icon={<SaveOutlined />} loading={savingAsset} onClick={() => void handleSaveAsset(false)}>
                                保存
                              </Button>
                              <Button size="small" type="primary" icon={<ReloadOutlined />} loading={savingAsset} onClick={() => void handleSaveAsset(true)}>
                                保存并重生成
                              </Button>
                              <Button size="small" onClick={() => setEditingAssetId(null)}>取消</Button>
                            </div>
                          </div>
                        ) : (
                          <div className="asset-mini-content">
                            <div className="asset-mini-title-row">
                              <strong>{item.name}</strong>
                              <Button size="small" icon={<EditOutlined />} onClick={() => startEditAsset(item)}>
                                编辑
                              </Button>
                            </div>
                            <span>{assetTab === 'characters' ? (item.personality || '性格待补充') : (item.description || '场景描述待补充')}</span>
                            {assetTab === 'characters' ? (
                              <>
                                <em>音色：{item.voice_id || 'Mimo 默认音色'}</em>
                                <em>人设：{item.appearance?.description || item.visual_prompt || '待补充'}</em>
                              </>
                            ) : (
                              <em>{item.visual_prompt || '视觉提示词待补充'}</em>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                    {!assetItems.length && <div className="asset-empty-card">素材生成后会在这里预览</div>}
                  </div>
                </div>
              </div>
            ) : (
              <div className="asset-empty-state">素材生成后会在这里集中管理</div>
            )}
          </div>
        )}

        {workspaceTab === 'av' && (
          <div className="av-panel panel-enter" {...getWorkspacePanelAriaProps('av')}>
            <Suspense fallback={<div className="av-loading">字幕与音频工作台加载中…</div>}>
              <AvWorkbench />
            </Suspense>
          </div>
        )}

        {showPreviewSurface && (
          <>
            {workspaceTab === 'review' && (
              <div className="review-command-bar panel-enter" role="region" aria-label="分镜审核操作">
                <div className="review-shot-state">
                  <span>当前镜头 {selectedShot ? String(selectedShot.sequence || 1).padStart(2, '0') : '--'}</span>
                  <strong className={selectedShot?.confirmed ? 'approved' : 'pending'}>
                    {selectedShot?.confirmed ? '已过审' : '待审核'}
                  </strong>
                </div>
                <div className="review-actions">
                  <Button
                    type="primary"
                    className="review-approve-main"
                    icon={<CheckCircleOutlined />}
                    disabled={!selectedShot || !selectedShotReady || selectedShot.confirmed}
                    onClick={() => selectedShot && void handleApproveShot(selectedShot.id, true)}
                  >
                    {selectedShot?.confirmed ? '已通过审批' : '通过当前镜头'}
                  </Button>
                  <Button
                    disabled={!selectedShot || !selectedShotReady}
                    onClick={() => selectedShot && void handleApproveShot(selectedShot.id, false)}
                  >
                    退回调整
                  </Button>
                  <Button
                    type="primary"
                    className="shot-video-action"
                    loading={confirming}
                    disabled={!selectedShot || !selectedShot.confirmed}
                    onClick={handleGenerateSelectedShotVideo}
                  >
                    {selectedShot?.video_path ? '重新生成本镜头' : '生成本镜头视频'}
                  </Button>
                </div>
              </div>
            )}

            {workspaceTab === 'review' && selectedShot && (
              <QualityReviewPanel shot={selectedShot} />
            )}

            <div
              className="preview-panel panel-enter"
              {...getWorkspacePanelAriaProps(workspaceTab)}
            >
        <div
          className={`preview-stage${dragStart ? ' dragging' : ''}`}
          onMouseDown={beginPreviewDrag}
          onMouseMove={movePreviewDrag}
          onMouseUp={() => setDragStart(null)}
          onMouseLeave={() => setDragStart(null)}
        >
          {workspaceTab === 'review' && storyboardReviewVisible && (
            <div className="confirm-overlay">
              <Button
                type="primary"
                size="small"
                className="shot-video-action"
                icon={<CheckCircleOutlined />}
                loading={confirming}
                disabled={!selectedShot || !selectedShot.confirmed}
                onClick={handleGenerateSelectedShotVideo}
            >
                {selectedShot?.video_path ? '重新生成本镜头' : selectedShot?.confirmed ? '生成本镜头视频' : `审核后生成 ${approvedShotCount}/${shots.length}`}
              </Button>
              {allShotVideosReady && (
                <Button
                  type="primary"
                  size="small"
                  className="shot-video-action"
                  loading={composing}
                  onClick={handleComposeProjectVideo}
                >
                  合成成片
                </Button>
              )}
            </div>
          )}

          {imageUrl && previewMode === 'shot' && !isGenerating && (
            <div className="preview-tools" aria-label="分镜预览工具" onMouseDown={(event) => event.stopPropagation()}>
              <button type="button" onClick={(event) => { event.stopPropagation(); changePreviewScale(-0.1) }} aria-label="缩小">
                <ZoomOutOutlined />
              </button>
              <span>{Math.round(previewScale * 100)}%</span>
              <button type="button" onClick={(event) => { event.stopPropagation(); changePreviewScale(0.1) }} aria-label="放大">
                <ZoomInOutlined />
              </button>
              <button type="button" onClick={(event) => { event.stopPropagation(); resetPreviewTransform() }} aria-label="重置视图">
                <ReloadOutlined />
              </button>
              <DragOutlined />
            </div>
          )}

          {/* 参数已修改：旧素材保留预览，但给出明确的「待重新生成」提示。 */}
          {!isGenerating && imageUrl && selectedShot?.media_stale && (
            <div className="media-stale-overlay" role="status" aria-live="polite">
              <span><ClockCircleOutlined /> 参数已修改，素材待重新生成（当前为旧素材）</span>
            </div>
          )}

          {currentVideoUrl && (
            <div className="preview-mode-switch" aria-label="预览模式">
              <button
                type="button"
                className={previewMode === 'shot' ? 'active' : ''}
                onClick={() => setPreviewMode('shot')}
              >
                分镜
              </button>
              <button
                type="button"
                className={previewMode === 'video' ? 'active' : ''}
                onClick={() => setPreviewMode('video')}
              >
                视频
              </button>
            </div>
          )}

          {isGenerating ? (
            <div className="preview-loading">
              <div>{getStepLabel(currentStep)}...</div>
              <div className="preview-loading-bar" />
            </div>
          ) : currentVideoUrl && previewMode === 'video' ? (
            <video
              src={currentVideoUrl}
              controls
              className="final-video"
              poster={imageUrl || undefined}
            />
          ) : imageUrl ? (
            <div className="preview-image-pan">
              <img
                src={imageUrl}
                alt="当前镜头预览"
                draggable={false}
                onClick={(event) => {
                  event.stopPropagation()
                  setImagePreview({ url: imageUrl, title: `镜头 ${selectedShot?.sequence || ''} 高清预览` })
                }}
                style={{
                  transform: `translate(${previewOffset.x}px, ${previewOffset.y}px) scale(${previewScale})`,
                }}
              />
            </div>
          ) : (
            <span className="preview-placeholder">选择镜头即可预览画面</span>
          )}

          {selectedShot && !isGenerating && previewMode === 'shot' && (
            <div className="preview-caption">
              <div className="preview-scene">{selectedShot.scene_description}</div>
              {selectedShot.dialogue && <div className="preview-dialogue">“{selectedShot.dialogue}”</div>}
              {workspaceTab === 'review' && selectedShotReady && (
                <div className="preview-approval-actions">
                  <Button
                    size="small"
                    className="approval-pass-btn"
                    type={selectedShot.confirmed ? 'primary' : 'default'}
                    onClick={() => void handleApproveShot(selectedShot.id, true)}
                    icon={<CheckCircleOutlined />}
                  >
                    通过此镜头
                  </Button>
                  <Button size="small" onClick={() => void handleApproveShot(selectedShot.id, false)}>
                    退回调整
                  </Button>
                  <span>{selectedShot.confirmed ? '已通过' : '待审核'}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {shots.length > 0 && (
          <>
          <div className="selective-regen-toolbar" role="region" aria-label="选择性重生成">
            <div className="selective-regen-heading">
              <strong>选择性重生成</strong>
              <span>{queueSelectedIds.length ? `已选 ${queueSelectedIds.length} 个镜头` : '点击缩略图左上角进行多选'}</span>
            </div>
            <Select
              size="small"
              mode="multiple"
              maxTagCount={1}
              value={queueStages}
              options={[
                { value: 'storyboard', label: '故事板' },
                { value: 'audio', label: '配音' },
                { value: 'video', label: '视频' },
              ]}
              onChange={(value: RegenerationQueueStage[]) => setQueueStages(value)}
              placeholder="选择阶段"
              aria-label="重生成阶段"
            />
            <Select
              size="small"
              value={queuePriority}
              onChange={setQueuePriority}
              options={[{ value: 5, label: '高优先级' }, { value: 0, label: '普通优先级' }, { value: -5, label: '低优先级' }]}
              aria-label="任务优先级"
            />
            <Select
              size="small"
              value={queueConcurrency}
              onChange={setQueueConcurrency}
              options={[1, 2, 3, 4].map((value) => ({ value, label: `${value} 路并发` }))}
              aria-label="并发数"
            />
            <Select
              size="small"
              value={queueOrder}
              onChange={setQueueOrder}
              options={[{ value: 'sequence', label: '按镜头顺序' }, { value: 'shot', label: '按选择顺序' }, { value: 'reverse', label: '倒序执行' }]}
              aria-label="执行顺序"
            />
            <Input
              size="small"
              type="number"
              min={1}
              value={queueVersion || ''}
              onChange={(event) => setQueueVersion(Math.max(0, Number(event.target.value) || 0))}
              placeholder="版本"
              aria-label="按镜头版本生成"
              style={{ width: 76 }}
            />
            <label className="selective-regen-check"><input type="checkbox" checked={queueReuseAudio} onChange={(event) => setQueueReuseAudio(event.target.checked)} />复用有效配音</label>
            <label className="selective-regen-check"><input type="checkbox" checked={queueResumeMissing} onChange={(event) => setQueueResumeMissing(event.target.checked)} />只补缺失产物</label>
            <label className="selective-regen-check selective-regen-danger"><input type="checkbox" checked={queueForceConfirmed} onChange={(event) => setQueueForceConfirmed(event.target.checked)} />强制确认镜头</label>
            <Button type="primary" size="small" loading={queueSubmitting} disabled={!queueSelectedIds.length} onClick={() => void submitSelectiveRegeneration()}>
              加入队列
            </Button>
          </div>
          <div className="thumb-strip">
            {shots.map((shot, i) => {
              const thumbUrl = toOutputUrl(shot.storyboard_path || shot.image_path)
              const isSelected = (selectedShotId || shots[0]?.id) === shot.id
              const isQueuedSelected = queueSelectedIds.includes(shot.id)

              return (
                <div
                  key={shot.id}
                  className={`thumb-item${isSelected ? ' active' : ''}${shot.confirmed ? ' approved' : ''}${isQueuedSelected ? ' queue-selected' : ''}${shot.media_stale ? ' media-stale' : ''}`}
                  onClick={() => openShotConfig(shot.id)}
                >
                  <button
                    type="button"
                    className="thumb-select-toggle"
                    aria-label={`${isQueuedSelected ? '取消选择' : '选择'}镜头 ${i + 1}`}
                    aria-pressed={isQueuedSelected}
                    onClick={(event) => { event.stopPropagation(); toggleQueueShot(shot.id) }}
                  >
                    {isQueuedSelected ? '✓' : ''}
                  </button>
                  {shot.media_stale && (
                    <span className="thumb-stale-badge" title="参数已修改，素材待重新生成">待重生成</span>
                  )}
                  {(() => {
                    const badge = qualityBadgeFor(shot.quality_review)
                    if (!badge) return null
                    return (
                      <span className={`thumb-quality-badge ${badge.className}`} title={badge.title}>
                        {badge.label}
                      </span>
                    )
                  })()}
                  {thumbUrl ? (
                    <img src={thumbUrl} alt={`镜头 ${i + 1}`} loading="lazy" decoding="async" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  ) : (
                    <div className="thumb-index">{i + 1}</div>
                  )}
                  <span className={`thumb-approval-badge${shot.confirmed ? ' approved' : ' pending'}`}>
                    {shot.confirmed ? <CheckCircleOutlined /> : <ClockCircleOutlined />}
                    <span>{shot.confirmed ? '已通过' : '待审核'}</span>
                  </span>
                </div>
              )
            })}
          </div>
          </>
        )}
            </div>
          </>
        )}
      </div>
      <Modal
        open={Boolean(imagePreview)}
        title={imagePreview?.title || '高清预览'}
        footer={null}
        centered
        width="min(1120px, 92vw)"
        className="image-preview-modal"
        onCancel={() => setImagePreview(null)}
      >
        {imagePreview && <img src={imagePreview.url} alt={imagePreview.title} />}
      </Modal>
      {estimateGate.modal}
    </section>
  )
}

export default MainWorkspace
