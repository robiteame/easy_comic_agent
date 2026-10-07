import type React from 'react'
import { useEffect } from 'react'
import Button from 'antd/es/button'
import Progress from 'antd/es/progress'
import { useUpdateStore } from '../stores/updateStore'
import { buildUpdateBannerView } from './updaterViewModel'

// 应用级更新横幅:主进程 updater 状态机的唯一非阻断展示位。
// 失败状态在这里只会降级为「前往下载」,绝不渲染错误弹窗。
const UpdateBanner: React.FC = () => {
  const state = useUpdateStore((store) => store.state)
  const mode = useUpdateStore((store) => store.mode)
  const dismissedKeys = useUpdateStore((store) => store.dismissedKeys)
  const initialize = useUpdateStore((store) => store.initialize)
  const runBannerAction = useUpdateStore((store) => store.runBannerAction)

  useEffect(() => {
    initialize()
  }, [initialize])

  const view = buildUpdateBannerView(state, mode, dismissedKeys)
  if (!view.visible) return null

  return (
    <div className={`update-banner update-banner-${view.level}`} role="status" aria-label="软件更新">
      <div className="update-banner-main">
        <div className="update-banner-title">{view.title}</div>
        <div className="update-banner-desc">
          {view.description}
          {view.note ? <span className="update-banner-note">{view.note}</span> : null}
        </div>
        {typeof view.percent === 'number' ? (
          <div className="update-banner-progress">
            <Progress
              percent={view.percent}
              size="small"
              status="active"
              showInfo={false}
              aria-label={`下载进度 ${view.percent}%`}
            />
            <span className="update-banner-percent">{`${view.percent.toFixed(1)}%`}</span>
          </div>
        ) : null}
      </div>
      <div className="update-banner-actions">
        {view.showReleaseLink ? (
          <Button size="small" type="link" onClick={() => runBannerAction('open-release-page')}>
            查看更新日志
          </Button>
        ) : null}
        {view.actions.map((action) => (
          <Button
            key={action.kind}
            size="small"
            type={action.kind === 'dismiss' ? 'text' : 'primary'}
            onClick={() => runBannerAction(action.kind)}
          >
            {action.label}
          </Button>
        ))}
      </div>
    </div>
  )
}

export default UpdateBanner
