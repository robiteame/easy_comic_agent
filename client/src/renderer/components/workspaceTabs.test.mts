import assert from 'node:assert/strict'
import { test } from 'vitest'
import {
  getWorkspacePanelAriaProps,
  getWorkspaceTabAriaProps,
  resolveWorkspaceTabFocus,
  resolveWorkspaceTabIndex,
} from './workspaceTabs.ts'

const TAB_IDS = ['script', 'assets', 'storyboard', 'review', 'video'] as const

test('每个标签的 aria-controls 都指向同 id 的 tabpanel，panel 的 aria-labelledby 反向指回', () => {
  TAB_IDS.forEach((tabId, index) => {
    const selectedId = TAB_IDS[index]
    const tab = getWorkspaceTabAriaProps(tabId, selectedId)
    const panel = getWorkspacePanelAriaProps(tabId)

    assert.equal(tab.role, 'tab')
    assert.equal(panel.role, 'tabpanel')
    assert.equal(tab['aria-controls'], panel.id, `${tabId}: aria-controls 必须等于 panel 的 id`)
    assert.equal(panel['aria-labelledby'], tab.id, `${tabId}: aria-labelledby 必须指回 tab 的 id`)
    assert.equal(tab['aria-selected'], true)
    assert.equal(tab.tabIndex, 0)
    assert.equal(panel.tabIndex, 0)
  })
})

test('roving tabindex：任意时刻只有一个标签进入 Tab 序列，且与 aria-selected 一致', () => {
  TAB_IDS.forEach((selectedId) => {
    const tabs = TAB_IDS.map((tabId) => getWorkspaceTabAriaProps(tabId, selectedId))
    const focusable = tabs.filter((tab) => tab.tabIndex === 0)
    const selected = tabs.filter((tab) => tab['aria-selected'])

    assert.equal(focusable.length, 1, `选中 ${selectedId} 时只能有一个 tabIndex=0`)
    assert.equal(selected.length, 1)
    assert.equal(focusable[0].id, selected[0].id, 'tabIndex=0 的标签必须就是 aria-selected 的标签')
    tabs
      .filter((tab) => tab.id !== selected[0].id)
      .forEach((tab) => {
        assert.equal(tab.tabIndex, -1, '未选中标签必须 tabIndex=-1')
      })
  })
})

test('方向键在标签间移动并在首尾循环', () => {
  assert.equal(resolveWorkspaceTabIndex('ArrowRight', 0, 5), 1)
  assert.equal(resolveWorkspaceTabIndex('ArrowRight', 4, 5), 0, '末尾右移回到第一个标签')
  assert.equal(resolveWorkspaceTabIndex('ArrowLeft', 0, 5), 4, '首个标签左移回到末尾')
  assert.equal(resolveWorkspaceTabIndex('ArrowLeft', 3, 5), 2)
  assert.equal(resolveWorkspaceTabFocus('ArrowRight', TAB_IDS, 'video'), 'script')
  assert.equal(resolveWorkspaceTabFocus('ArrowLeft', TAB_IDS, 'script'), 'video')
})

test('Home / End 跳到首尾标签', () => {
  assert.equal(resolveWorkspaceTabIndex('Home', 3, 5), 0)
  assert.equal(resolveWorkspaceTabIndex('End', 1, 5), 4)
  assert.equal(resolveWorkspaceTabFocus('Home', TAB_IDS, 'review'), 'script')
  assert.equal(resolveWorkspaceTabFocus('End', TAB_IDS, 'script'), 'video')
})

test('只拦截方向键与 Home / End，其他按键交给浏览器默认行为', () => {
  ;['Enter', ' ', 'Tab', 'Escape', 'ArrowUp', 'ArrowDown', 'a'].forEach((key) => {
    assert.equal(resolveWorkspaceTabIndex(key, 2, 5), null, `${key} 不应被标签页拦截`)
    assert.equal(resolveWorkspaceTabFocus(key, TAB_IDS, 'storyboard'), null)
  })
  assert.equal(resolveWorkspaceTabIndex('ArrowRight', 0, 0), null, '空标签列表不应产生目标下标')
  assert.equal(resolveWorkspaceTabIndex('ArrowRight', 9, 5), null, '越界下标不应产生目标下标')
})
