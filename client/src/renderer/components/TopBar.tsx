import type React from 'react'

export const OPEN_SETTINGS_EVENT = 'workspace:open-settings'

const TopBar: React.FC = () => {
  // macOS 使用 hiddenInset 无边框窗口，红绿灯悬浮在顶部拖拽条上，需要更高的
  // 非交互区域；Windows 保留原生标题栏，维持原 10px 条即可。
  const isMac = navigator.userAgent.includes('Macintosh')
  return <header className={isMac ? 'topbar topbar-mac' : 'topbar'} />
}

export default TopBar
