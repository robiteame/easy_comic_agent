/**
 * 工作区标签页的纯逻辑：DOM id、ARIA 属性与键盘导航。
 * 抽离成无 DOM 依赖的纯函数，便于在 node:test 下直接覆盖标签页交互。
 */

export const WORKSPACE_TAB_ID_PREFIX = 'workspace-tab-'
export const WORKSPACE_PANEL_ID_PREFIX = 'workspace-panel-'

export interface WorkspaceTabAriaProps {
  id: string
  role: 'tab'
  'aria-selected': boolean
  'aria-controls': string
  tabIndex: 0 | -1
}

export interface WorkspacePanelAriaProps {
  id: string
  role: 'tabpanel'
  'aria-labelledby': string
  tabIndex: 0
}

export function workspaceTabDomId(tabId: string): string {
  return `${WORKSPACE_TAB_ID_PREFIX}${tabId}`
}

export function workspacePanelDomId(tabId: string): string {
  return `${WORKSPACE_PANEL_ID_PREFIX}${tabId}`
}

/** 自动激活模式：只有选中的标签 tabIndex=0，其余为 -1（roving tabindex）。 */
export function getWorkspaceTabAriaProps(tabId: string, selectedTabId: string): WorkspaceTabAriaProps {
  const selected = tabId === selectedTabId
  return {
    id: workspaceTabDomId(tabId),
    role: 'tab',
    'aria-selected': selected,
    'aria-controls': workspacePanelDomId(tabId),
    tabIndex: selected ? 0 : -1,
  }
}

/** 面板用 aria-labelledby 反向指回自己的标签，保证 id 成对、不会指向不存在的元素。 */
export function getWorkspacePanelAriaProps(tabId: string): WorkspacePanelAriaProps {
  return {
    id: workspacePanelDomId(tabId),
    role: 'tabpanel',
    'aria-labelledby': workspaceTabDomId(tabId),
    tabIndex: 0,
  }
}

/**
 * 方向键 / Home / End 的目标下标（自动激活模式，首尾循环）。
 * 其它按键返回 null，调用方不得拦截，以免破坏常规键盘操作。
 */
export function resolveWorkspaceTabIndex(key: string, currentIndex: number, tabCount: number): number | null {
  if (tabCount <= 0) return null
  if (currentIndex < 0 || currentIndex >= tabCount) return null
  switch (key) {
    case 'ArrowRight':
      return (currentIndex + 1) % tabCount
    case 'ArrowLeft':
      return (currentIndex - 1 + tabCount) % tabCount
    case 'Home':
      return 0
    case 'End':
      return tabCount - 1
    default:
      return null
  }
}

/** 返回按键后应聚焦并选中的标签 id；按键未被处理时返回 null。 */
export function resolveWorkspaceTabFocus<TabId extends string>(
  key: string,
  tabIds: readonly TabId[],
  currentTabId: TabId,
): TabId | null {
  const foundIndex = tabIds.indexOf(currentTabId)
  const nextIndex = resolveWorkspaceTabIndex(key, foundIndex >= 0 ? foundIndex : 0, tabIds.length)
  return nextIndex === null ? null : tabIds[nextIndex]
}
