import type React from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AppstoreAddOutlined,
  CaretDownOutlined,
  CaretRightOutlined,
  CloudUploadOutlined,
  ControlOutlined,
  DeleteOutlined,
  FolderOpenOutlined,
  LeftOutlined,
  PlaySquareOutlined,
  ProjectOutlined,
  RightOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons'
import message from 'antd/es/message'
import Modal from 'antd/es/modal'
import Tooltip from 'antd/es/tooltip'
import { characterApi, projectApi, shotApi } from '../services/api'
import { PROJECTS_REFRESHED_EVENT } from '../constants/events'
import { beginProjectNavigationIntent, requestProjectNavigation } from '../services/projectNavigationGuard'
import { useProjectStore } from '../stores/projectStore'
import { useShotStore } from '../stores/shotStore'
import { useTaskStore } from '../stores/taskStore'
import { OPEN_TASK_CENTER_EVENT } from './TaskCenter'
import { OPEN_SETTINGS_EVENT } from './TopBar'

const OPEN_CREATE_PROJECT_EVENT = 'workspace:open-create-project'
const WORKSPACE_NAVIGATE_EVENT = 'workspace:navigate'
const OPEN_PROJECT_EVENT = 'workspace:open-project'

interface ProjectItem {
  id: string
  parent_project_id?: string
  project_type?: 'series' | 'episode'
  episode_number?: number
  title: string
  status: string
  style?: string
  genre?: string
  updated_at?: string
  video_path?: string
}

interface LeftSidebarProps {
  collapsed: boolean
  onToggleCollapsed: () => void
}

function formatUpdatedAt(value?: string) {
  if (!value) return '刚刚创建'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '刚刚创建'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function cleanTitle(value?: string) {
  const title = (value || '').trim()
  return title || '未命名项目'
}

const LeftSidebar: React.FC<LeftSidebarProps> = ({ collapsed, onToggleCollapsed }) => {
  const { projectId, parentProjectId, projectType, style, outputFormat, resolution, platform, setProject, reset } =
    useProjectStore()
  const {
    setShots,
    selectShot,
    setGenerating,
    setProgress,
    setAwaitingStoryboardConfirm,
    setVideoPath,
    appendLog,
    clearLogs,
  } = useShotStore()
  const taskSummary = useTaskStore((state) => state.summary)

  const [projects, setProjects] = useState<ProjectItem[]>([])
  const [expandedSeriesIds, setExpandedSeriesIds] = useState<Set<string>>(new Set())
  const [loadingProjectId, setLoadingProjectId] = useState<string | null>(null)
  const finalVideoInputRef = useRef<HTMLInputElement | null>(null)
  const projectSelectionRequestRef = useRef(0)
  const projectListRequestRef = useRef(0)
  const importVideoRequestRef = useRef(0)
  const createEpisodeRequestRef = useRef(0)
  const projectContextEpochRef = useRef(0)
  const mountedRef = useRef(false)

  const refreshProjects = () => {
    const requestId = ++projectListRequestRef.current
    projectApi
      .list()
      .then((nextProjects) => {
        if (mountedRef.current && requestId === projectListRequestRef.current) setProjects(nextProjects)
      })
      .catch(() => undefined)
  }

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      projectSelectionRequestRef.current += 1
      projectListRequestRef.current += 1
      importVideoRequestRef.current += 1
      createEpisodeRequestRef.current += 1
    }
  }, [])

  useEffect(() => {
    refreshProjects()
    projectContextEpochRef.current += 1
  }, [projectId])

  // 后端改了项目标题（如剧本解析按剧本自动命名）后刷新项目列表，保证名称实时。
  useEffect(() => {
    const refresh = () => refreshProjects()
    window.addEventListener(PROJECTS_REFRESHED_EVENT, refresh)
    return () => window.removeEventListener(PROJECTS_REFRESHED_EVENT, refresh)
  }, [])

  // 任务中心的「跳转到对应项目」：复用项目切换的全部守卫（导航意图 + 请求竞态）。
  useEffect(() => {
    const openProject = (event: Event) => {
      const detail = (event as CustomEvent<{ projectId?: string; shotId?: string }>).detail || {}
      if (!detail.projectId) return
      void handleSelectProject(detail.projectId).then(() => {
        if (detail.shotId) {
          window.dispatchEvent(
            new CustomEvent('workspace:open-shot', { detail: { projectId: detail.projectId, shotId: detail.shotId } }),
          )
        }
      })
    }
    window.addEventListener(OPEN_PROJECT_EVENT, openProject)
    return () => window.removeEventListener(OPEN_PROJECT_EVENT, openProject)
  })

  const visibleProjects = useMemo(() => {
    return [...projects].sort((a, b) => {
      const ta = a.updated_at ? new Date(a.updated_at).getTime() : 0
      const tb = b.updated_at ? new Date(b.updated_at).getTime() : 0
      return tb - ta
    })
  }, [projects])

  const projectTree = useMemo(() => {
    const series = visibleProjects.filter((item) => (item.project_type || 'series') !== 'episode')
    const episodesByParent = new Map<string, ProjectItem[]>()

    visibleProjects
      .filter((item) => item.project_type === 'episode')
      .forEach((episode) => {
        const key = episode.parent_project_id || ''
        episodesByParent.set(key, [...(episodesByParent.get(key) || []), episode])
      })

    episodesByParent.forEach((items) => {
      items.sort((a, b) => (a.episode_number || 0) - (b.episode_number || 0))
    })

    return { series, episodesByParent }
  }, [visibleProjects])

  useEffect(() => {
    const activeRootId = projectType === 'episode' ? parentProjectId : projectId
    if (!activeRootId) return
    setExpandedSeriesIds((prev) => new Set(prev).add(activeRootId))
  }, [parentProjectId, projectId, projectType])

  const handleSelectProject = async (id: string) => {
    beginProjectNavigationIntent()
    const requestId = ++projectSelectionRequestRef.current
    try {
      const localProject = projects.find((item) => item.id === id)
      if (localProject && (localProject.project_type || 'series') !== 'episode') {
        const firstEpisode = (projectTree.episodesByParent.get(id) || [])[0]
        if (firstEpisode) {
          setExpandedSeriesIds((prev) => new Set(prev).add(id))
          await handleSelectProject(firstEpisode.id)
          return
        }
      }
      setLoadingProjectId(id)

      const currentProjectId = useProjectStore.getState().projectId
      if (id !== currentProjectId) {
        const canNavigate = await requestProjectNavigation(currentProjectId, id)
        if (
          !canNavigate ||
          requestId !== projectSelectionRequestRef.current ||
          useProjectStore.getState().projectId !== currentProjectId
        ) {
          return
        }
      }

      // Hide the previous project's data immediately. The workspace performs
      // the same reset when the project store changes; doing it here avoids a
      // stale frame during the request gap.
      if (id !== useProjectStore.getState().projectId) {
        // Invalidate the current workspace before requests for the next
        // project resolve. This also closes the old project's WebSocket.
        setProject({
          projectId: null,
          parentProjectId: '',
          parentProjectTitle: '',
          projectType: 'series',
          episodeNumber: 0,
          title: '未命名项目',
          status: 'draft',
          characters: [],
        })
        setShots([])
        selectShot(null)
        setGenerating(false)
        setAwaitingStoryboardConfirm(false)
        setVideoPath('')
        setProgress(0, '')
        clearLogs()
      }

      const [projectDetail, shotList, characters] = await Promise.all([
        projectApi.get(id),
        shotApi.list(id),
        characterApi.list(id),
      ])

      // A later click wins. Ignore all late responses from an earlier
      // selection so they cannot overwrite the active project's state.
      if (requestId !== projectSelectionRequestRef.current) return
      const expectedProjectId = id === currentProjectId ? currentProjectId : null
      if (useProjectStore.getState().projectId !== expectedProjectId) return

      setProject({
        projectId: projectDetail.id,
        parentProjectId: projectDetail.parent_project_id || '',
        parentProjectTitle: projectDetail.parent_project_title || '',
        projectType: projectDetail.project_type || 'series',
        episodeNumber: projectDetail.episode_number || 0,
        title: projectDetail.title,
        genre: projectDetail.genre,
        style: projectDetail.style,
        status: projectDetail.status,
        outputFormat: projectDetail.output_format,
        resolution: projectDetail.resolution,
        platform: projectDetail.platform,
        characters,
      })

      setShots(shotList || [])
      selectShot(shotList?.[0]?.id || null)
      setAwaitingStoryboardConfirm(projectDetail.status === 'storyboard_ready')
      setVideoPath(
        projectDetail.video_path ||
          (projectDetail.status === 'completed' ? `/output/projects/${projectDetail.id}/output/final.mp4` : ''),
      )
      window.dispatchEvent(new CustomEvent(WORKSPACE_NAVIGATE_EVENT, { detail: { tab: 'script' } }))
    } catch (err: any) {
      if (requestId !== projectSelectionRequestRef.current) return
      message.error('加载项目失败：' + (err.message || '未知错误'))
    } finally {
      if (requestId === projectSelectionRequestRef.current) setLoadingProjectId(null)
    }
  }

  const handleImportFinalVideo = async (file: File) => {
    if (!projectId) {
      message.warning('请先选择或创建一个剧集')
      return
    }
    if (projectType !== 'episode') {
      message.warning('请先选择具体剧集，再导入成片')
      return
    }

    const entryProjectId = projectId
    const requestId = ++importVideoRequestRef.current
    const projectEpoch = projectContextEpochRef.current

    try {
      setGenerating(true)
      setProgress(95, 'quality_check')
      const formData = new FormData()
      formData.append('file', file)
      const result = await projectApi.importVideo(entryProjectId, formData)
      if (
        requestId !== importVideoRequestRef.current ||
        projectEpoch !== projectContextEpochRef.current ||
        useProjectStore.getState().projectId !== entryProjectId
      )
        return
      setVideoPath(result.video_path || `/output/projects/${entryProjectId}/output/final.mp4`)
      appendLog(`[${new Date().toLocaleTimeString('zh-CN', { hour12: false })}] 已导入成片：${file.name}`)
      refreshProjects()
      message.success('成片已导入当前剧集')
    } catch (err: any) {
      if (
        requestId !== importVideoRequestRef.current ||
        projectEpoch !== projectContextEpochRef.current ||
        useProjectStore.getState().projectId !== entryProjectId
      )
        return
      message.error('成片导入失败：' + (err.message || '未知错误'))
    } finally {
      if (
        requestId === importVideoRequestRef.current &&
        projectEpoch === projectContextEpochRef.current &&
        useProjectStore.getState().projectId === entryProjectId
      ) {
        setGenerating(false)
      }
    }
  }

  const handleCreateEpisode = async () => {
    beginProjectNavigationIntent()
    if (!projectId) {
      message.warning('请先选择一个大项目')
      return
    }
    const rootId = projectType === 'episode' ? parentProjectId : projectId
    if (!rootId) {
      message.warning('请先选择一个大项目')
      return
    }

    const sourceProjectId = projectId
    const requestId = ++createEpisodeRequestRef.current
    const projectEpoch = projectContextEpochRef.current

    try {
      const canNavigate = await requestProjectNavigation(sourceProjectId, null)
      if (
        !canNavigate ||
        requestId !== createEpisodeRequestRef.current ||
        projectEpoch !== projectContextEpochRef.current ||
        useProjectStore.getState().projectId !== sourceProjectId
      ) {
        return
      }
      const nextNumber = visibleProjects.filter((item) => item.parent_project_id === rootId).length + 1
      projectListRequestRef.current += 1
      const episode = await projectApi.create({
        title: `第 ${nextNumber} 集`,
        parent_project_id: rootId,
        project_type: 'episode',
        episode_number: nextNumber,
        style,
        genre: '',
        output_format: outputFormat,
        resolution,
        platform,
      })
      setExpandedSeriesIds((prev) => new Set(prev).add(rootId))
      refreshProjects()
      if (
        requestId !== createEpisodeRequestRef.current ||
        projectEpoch !== projectContextEpochRef.current ||
        useProjectStore.getState().projectId !== sourceProjectId
      )
        return
      await handleSelectProject(episode.id)
      if (useProjectStore.getState().projectId === episode.id) {
        message.success('单集已创建')
      }
    } catch (err: any) {
      if (
        requestId !== createEpisodeRequestRef.current ||
        projectEpoch !== projectContextEpochRef.current ||
        useProjectStore.getState().projectId !== sourceProjectId
      )
        return
      message.error('创建单集失败：' + (err.message || '未知错误'))
    }
  }

  const handleDeleteProject = (project: ProjectItem, event: React.MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()
    const isEpisode = project.project_type === 'episode'
    const title = cleanTitle(project.title)
    Modal.confirm({
      title: `删除${isEpisode ? '单集' : '项目'}：${title}`,
      content: isEpisode
        ? '删除后会同步清理该单集下的分镜、故事板、成片和输出资产文件。'
        : '删除后会同步删除大项目、所有单集，以及绑定的角色板、场景板、分镜、成片和输出资产文件。',
      okText: '确认删除',
      cancelText: '取消',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          beginProjectNavigationIntent()
          projectSelectionRequestRef.current += 1
          const currentBeforeDelete = useProjectStore.getState()
          const deletingCurrent =
            currentBeforeDelete.projectId === project.id ||
            (project.project_type !== 'episode' && currentBeforeDelete.parentProjectId === project.id)
          if (deletingCurrent && !(await requestProjectNavigation(currentBeforeDelete.projectId, null))) {
            return
          }
          projectListRequestRef.current += 1
          await projectApi.delete(project.id)
          const currentProject = useProjectStore.getState()
          const deletedCurrent =
            project.id === currentProject.projectId ||
            (project.project_type !== 'episode' && currentProject.parentProjectId === project.id)
          if (deletedCurrent) {
            projectSelectionRequestRef.current += 1
            reset()
            setShots([])
            selectShot(null)
            setGenerating(false)
            setAwaitingStoryboardConfirm(false)
            setVideoPath('')
            setProgress(0, '')
            clearLogs()
          }
          refreshProjects()
          message.success('项目已删除，关联资产已清理')
        } catch (err: any) {
          message.error('项目删除失败：' + (err.response?.data?.detail || err.message || '未知错误'))
          throw err
        }
      },
    })
  }

  const toggleSeries = (id: string) => {
    setExpandedSeriesIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const renderProjectRow = (project: ProjectItem, isEpisode = false) => {
    const isActive = project.id === projectId
    const isLoading = loadingProjectId === project.id
    const title = cleanTitle(project.title)

    return (
      <div
        key={project.id}
        className={`project-item linear-project-item${isActive ? ' active' : ''}${isLoading ? ' loading' : ''}${isEpisode ? ' episode-item' : ''}`}
      >
        <button type="button" className="project-main" onClick={() => void handleSelectProject(project.id)}>
          <div className="project-item-title">
            <span className={`project-item-icon${isEpisode ? '' : ' folder'}`}>
              {isEpisode ? <PlaySquareOutlined /> : <FolderOpenOutlined />}
            </span>
            <span>
              {isEpisode ? `第 ${project.episode_number || 1} 集 · ` : ''}
              {title}
            </span>
          </div>
          <div className="project-item-meta">最近编辑：{formatUpdatedAt(project.updated_at)}</div>
        </button>
        <Tooltip title={isEpisode ? '删除单集' : '删除项目'}>
          <button
            type="button"
            className="project-delete-btn"
            aria-label={isEpisode ? '删除单集' : '删除项目'}
            onMouseDown={(event) => {
              event.preventDefault()
              event.stopPropagation()
            }}
            onPointerDown={(event) => event.stopPropagation()}
            onClick={(event) => handleDeleteProject(project, event)}
          >
            <DeleteOutlined />
          </button>
        </Tooltip>
      </div>
    )
  }

  const actionItems = [
    {
      id: 'create',
      label: '新建项目',
      icon: <AppstoreAddOutlined />,
      onClick: () => window.dispatchEvent(new CustomEvent(OPEN_CREATE_PROJECT_EVENT)),
    },
    { id: 'episode', label: '新建剧集', icon: <ProjectOutlined />, onClick: () => void handleCreateEpisode() },
    {
      id: 'import-video',
      label: '导入成片',
      icon: <CloudUploadOutlined />,
      onClick: () => finalVideoInputRef.current?.click(),
    },
  ]

  const openSettings = () => window.dispatchEvent(new CustomEvent(OPEN_SETTINGS_EVENT))
  const openTaskCenter = () => window.dispatchEvent(new CustomEvent(OPEN_TASK_CENTER_EVENT))
  const taskBadgeCount = taskSummary.activeCount + taskSummary.failedCount + taskSummary.interruptedCount
  const taskBadgeLabel =
    taskSummary.activeCount > 0
      ? `${taskSummary.activeCount} 个任务进行中`
      : taskSummary.failedCount > 0
        ? `${taskSummary.failedCount} 个任务失败`
        : taskSummary.latest
          ? `最近任务：${taskSummary.latest.status_label}`
          : '暂无任务'

  return (
    <aside className={`left-sidebar${collapsed ? ' collapsed' : ''}`} aria-label="左侧导航">
      <div className="sidebar-head">
        <button type="button" className="collapse-switch" onClick={onToggleCollapsed} aria-label="展开或收起左侧栏">
          {collapsed ? <RightOutlined /> : <LeftOutlined />}
        </button>
        {!collapsed && <div className="sidebar-brand">漫剧工坊</div>}
      </div>

      <div className="sidebar-content">
        <div className={collapsed ? 'collapsed-shortcuts' : 'sidebar-quick-actions linear-actions'}>
          {actionItems.map((item) =>
            collapsed ? (
              <Tooltip title={item.label} key={item.id} placement="right">
                <button type="button" className="collapsed-project-btn" onClick={item.onClick} aria-label={item.label}>
                  {item.icon}
                </button>
              </Tooltip>
            ) : (
              <div key={item.id} className="linear-row">
                <button
                  type="button"
                  className={`linear-action${item.id === 'create' ? ' primary' : ''}`}
                  onClick={item.onClick}
                >
                  <span className="linear-action-icon">{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              </div>
            ),
          )}

          <div className="project-list linear-project-list">
            {projectTree.series.length > 0 ? (
              projectTree.series.map((series) => {
                const title = cleanTitle(series.title)
                const episodes = projectTree.episodesByParent.get(series.id) || []
                const expanded = expandedSeriesIds.has(series.id)
                const isActive = series.id === projectId

                if (collapsed) {
                  return (
                    <Tooltip title={title} key={series.id} placement="right">
                      <button
                        type="button"
                        className={`collapsed-project-btn${isActive ? ' active' : ''}`}
                        onClick={() => void handleSelectProject(series.id)}
                        aria-label={title}
                      >
                        <FolderOpenOutlined />
                      </button>
                    </Tooltip>
                  )
                }

                return (
                  <div className="project-tree-node" key={series.id}>
                    <div className="project-series-row">
                      <button
                        type="button"
                        className="project-expand-btn"
                        onClick={() => toggleSeries(series.id)}
                        aria-label={expanded ? '收起剧集' : '展开剧集'}
                      >
                        {expanded ? <CaretDownOutlined /> : <CaretRightOutlined />}
                      </button>
                      {renderProjectRow(series)}
                    </div>
                    {expanded && episodes.map((episode) => renderProjectRow(episode, true))}
                  </div>
                )
              })
            ) : (
              <div className="empty-hint">暂无项目</div>
            )}
          </div>
        </div>
      </div>

      <div className="sidebar-footer">
        {collapsed ? (
          <>
            <Tooltip title={'任务中心 · ' + taskBadgeLabel} placement="right">
              <button
                type="button"
                className="collapsed-project-btn sidebar-task-center-btn"
                onClick={openTaskCenter}
                aria-label={'任务中心，' + taskBadgeLabel}
                aria-haspopup="dialog"
              >
                <UnorderedListOutlined />
                {taskBadgeCount > 0 && (
                  <span className="sidebar-task-badge" aria-hidden="true">
                    {taskBadgeCount}
                  </span>
                )}
              </button>
            </Tooltip>
            <Tooltip title="系统设置" placement="right">
              <button
                type="button"
                className="collapsed-project-btn sidebar-settings-btn"
                onClick={openSettings}
                aria-label="系统设置"
              >
                <ControlOutlined />
              </button>
            </Tooltip>
          </>
        ) : (
          <>
            <button
              type="button"
              className="linear-action sidebar-task-center-btn"
              onClick={openTaskCenter}
              aria-haspopup="dialog"
              aria-label={'任务中心，' + taskBadgeLabel}
            >
              <span className="linear-action-icon">
                <UnorderedListOutlined />
              </span>
              <span className="sidebar-task-copy">
                <span>任务中心</span>
                <em>{taskBadgeLabel}</em>
              </span>
              {taskBadgeCount > 0 && <span className="sidebar-task-badge">{taskBadgeCount}</span>}
            </button>
            <button type="button" className="linear-action sidebar-settings-btn" onClick={openSettings}>
              <span className="linear-action-icon">
                <ControlOutlined />
              </span>
              <span>系统设置</span>
            </button>
          </>
        )}
      </div>

      <input
        ref={finalVideoInputRef}
        type="file"
        accept="video/mp4,.mp4,.m4v"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) {
            void handleImportFinalVideo(file)
          }
          e.target.value = ''
        }}
      />
    </aside>
  )
}

export default LeftSidebar
