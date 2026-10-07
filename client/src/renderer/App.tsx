import React, { useEffect, useRef, useState } from 'react'
import BottomBar from './components/BottomBar'
import FirstRunWizard, { OPEN_WIZARD_EVENT } from './components/FirstRunWizard'
import GlobalPlayfulMotion from './components/GlobalPlayfulMotion'
import LeftSidebar from './components/LeftSidebar'
import MainWorkspace from './components/MainWorkspace'
import RightSidebar from './components/RightSidebar'
import TaskCenter, { OPEN_TASK_CENTER_EVENT } from './components/TaskCenter'
import TopBar from './components/TopBar'
import { OPEN_SETTINGS_EVENT } from './components/TopBar'
import UpdateBanner from './components/UpdateBanner'
import { settingsApi } from './services/api'
import { beginProjectNavigationIntent, requestProjectNavigation } from './services/projectNavigationGuard'
import { useProjectStore } from './stores/projectStore'
import { useTaskStore } from './stores/taskStore'

const OPEN_CREATE_PROJECT_EVENT = 'workspace:open-create-project'
const WORKSPACE_NAVIGATE_EVENT = 'workspace:navigate'
const SystemSettingsPage = React.lazy(() => import('./components/SystemSettingsPage'))

type WorkspaceNavigateDetail = {
  tab?: string
  previewMode?: 'shot' | 'video'
}

const App: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(true)
  const [settingsPageOpen, setSettingsPageOpen] = useState(false)
  const [taskCenterOpen, setTaskCenterOpen] = useState(false)
  const [wizardOpen, setWizardOpen] = useState(false)
  const settingsPageOpenRef = useRef(settingsPageOpen)
  const pendingWorkspaceNavigationRef = useRef<WorkspaceNavigateDetail | null>(null)
  const settingsNavigationRequestRef = useRef(0)

  useEffect(() => {
    settingsPageOpenRef.current = settingsPageOpen
  }, [settingsPageOpen])

  // 首启判定：读后端持久化标志（DATA_DIR 内，随安装迁移；不用 localStorage）。
  // 完成标志在后端写入，渲染进程只读；查询失败时保守不弹向导，避免后端启动
  // 竞态误伤第二次及以后启动。
  useEffect(() => {
    let active = true
    const check = (retried: boolean) => {
      settingsApi
        .onboardingStatus()
        .then((status) => {
          if (active && status && !status.completed) setWizardOpen(true)
        })
        .catch(() => {
          // 打包环境后端端口注入可能略晚于渲染进程首帧，失败重试一次。
          if (active && !retried) window.setTimeout(() => check(true), 1500)
        })
    }
    check(false)
    return () => {
      active = false
    }
  }, [])

  // 设置页「重新运行向导」入口：手动触发不重置完成标志，向导完成时照常回写。
  useEffect(() => {
    const openWizard = () => setWizardOpen(true)
    window.addEventListener(OPEN_WIZARD_EVENT, openWizard)
    return () => window.removeEventListener(OPEN_WIZARD_EVENT, openWizard)
  }, [])

  useEffect(() => {
    const openShotConfig = () => setRightSidebarCollapsed(false)
    window.addEventListener('workspace:open-shot-config', openShotConfig)
    return () => window.removeEventListener('workspace:open-shot-config', openShotConfig)
  }, [])

  // 任务中心连接在应用生命周期内保持：关闭面板不停止 WebSocket，任务自然继续
  // 在后台运行；卸载时统一清理 socket、定时器与事件缓冲。
  useEffect(() => {
    useTaskStore.getState().start()
    return () => {
      useTaskStore.getState().stop()
    }
  }, [])

  useEffect(() => {
    const openTaskCenter = () => setTaskCenterOpen(true)
    window.addEventListener(OPEN_TASK_CENTER_EVENT, openTaskCenter)
    return () => window.removeEventListener(OPEN_TASK_CENTER_EVENT, openTaskCenter)
  }, [])

  useEffect(() => {
    const openSettings = () => {
      const requestId = ++settingsNavigationRequestRef.current
      beginProjectNavigationIntent()
      const activeProjectId = useProjectStore.getState().projectId
      void requestProjectNavigation(activeProjectId, null).then((canNavigate) => {
        if (requestId !== settingsNavigationRequestRef.current || !canNavigate) return
        setSettingsPageOpen(true)
      })
    }
    window.addEventListener(OPEN_SETTINGS_EVENT, openSettings)
    return () => window.removeEventListener(OPEN_SETTINGS_EVENT, openSettings)
  }, [])

  useEffect(() => {
    const openCreateProject = () => {
      if (!settingsPageOpenRef.current) return
      settingsPageOpenRef.current = false
      setSettingsPageOpen(false)
      window.setTimeout(() => {
        window.dispatchEvent(new CustomEvent(OPEN_CREATE_PROJECT_EVENT))
      }, 50)
    }
    window.addEventListener(OPEN_CREATE_PROJECT_EVENT, openCreateProject)
    return () => window.removeEventListener(OPEN_CREATE_PROJECT_EVENT, openCreateProject)
  }, [])

  useEffect(() => {
    const navigateToWorkspace = (event: Event) => {
      const detail = (event as CustomEvent<WorkspaceNavigateDetail>).detail || {}
      if (!settingsPageOpenRef.current) return

      pendingWorkspaceNavigationRef.current = detail
      settingsPageOpenRef.current = false
      setSettingsPageOpen(false)
      window.setTimeout(() => {
        const pending = pendingWorkspaceNavigationRef.current
        pendingWorkspaceNavigationRef.current = null
        if (pending) {
          window.dispatchEvent(new CustomEvent(WORKSPACE_NAVIGATE_EVENT, { detail: pending }))
        }
      }, 50)
    }

    window.addEventListener(WORKSPACE_NAVIGATE_EVENT, navigateToWorkspace)
    return () => window.removeEventListener(WORKSPACE_NAVIGATE_EVENT, navigateToWorkspace)
  }, [])

  return (
    <div className="app-shell">
      <GlobalPlayfulMotion />
      <UpdateBanner />
      <TopBar />
      <div
        className={`app-main${sidebarCollapsed ? ' sidebar-collapsed' : ''}${rightSidebarCollapsed ? ' right-sidebar-collapsed' : ''}${settingsPageOpen ? ' settings-mode' : ''}`}
        role="main"
        aria-label="主工作台"
      >
        <LeftSidebar collapsed={sidebarCollapsed} onToggleCollapsed={() => setSidebarCollapsed((prev) => !prev)} />
        {settingsPageOpen ? (
          <React.Suspense
            fallback={
              <section className="main-workspace lazy-page-status" role="status">
                正在加载系统设置...
              </section>
            }
          >
            <SystemSettingsPage onBack={() => setSettingsPageOpen(false)} />
          </React.Suspense>
        ) : (
          <MainWorkspace />
        )}
        {!settingsPageOpen && (
          <RightSidebar
            collapsed={rightSidebarCollapsed}
            onToggleCollapsed={() => setRightSidebarCollapsed((prev) => !prev)}
          />
        )}
      </div>
      <BottomBar />
      <TaskCenter open={taskCenterOpen} onClose={() => setTaskCenterOpen(false)} />
      <FirstRunWizard open={wizardOpen} onClose={() => setWizardOpen(false)} />
    </div>
  )
}

export default App
