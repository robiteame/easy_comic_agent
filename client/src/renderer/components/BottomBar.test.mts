import assert from 'node:assert/strict'
import { test, beforeEach } from 'vitest'

import { expectRender } from '../test-support/ssrTestHelper.mts'
import { useShotStore } from '../stores/shotStore'
import { useTaskStore } from '../stores/taskStore'
import { useProjectStore } from '../stores/projectStore'
import BottomBar from './BottomBar'

function shotFixture(duration: number) {
  return {
    id: `shot-${duration}`,
    project_id: 'proj-1',
    sequence: 1,
    shot_type: 'normal',
    scene_description: '',
    character_action: '',
    dialogue: '',
    camera_angle: '',
    camera_movement: '',
    duration,
    emotion: '',
    transition: '',
    visual_notes: '',
    image_path: '',
    storyboard_path: '',
    video_path: '',
    audio_path: '',
    status: 'done',
    storyboard_status: 'approved',
    version: 1,
    confirmed: true,
    characters_in_scene: [],
    scene_asset_id: '',
    character_asset_ids: [],
  }
}

function resetStores() {
  useShotStore.setState({
    shots: [],
    isGenerating: false,
    progress: 0,
    currentStep: '',
    videoPath: '',
  })
  useTaskStore.setState({ jobs: [], connectionState: 'idle', summary: emptySummary() })
  useProjectStore.setState({ projectId: null })
}

function emptySummary() {
  return {
    activeCount: 0,
    failedCount: 0,
    interruptedCount: 0,
    retryableCount: 0,
    total: 0,
    latest: null,
    failedByCategory: [],
  }
}

beforeEach(resetStores)

test('空项目：显示等待输入剧本与 0% 进度条', () => {
  const html = expectRender(BottomBar)
  assert.ok(html.includes('等待输入剧本'), '应显示空态文案')
  assert.ok(html.includes('width:0%'), '空态进度条应为 0%')
})

test('已有镜头且空闲：显示总时长与 100% 进度', () => {
  useShotStore.setState({ shots: [shotFixture(4.5), shotFixture(3.5)] })
  const html = expectRender(BottomBar)
  assert.ok(html.includes('总时长 8.0 秒'), '总时长应按镜头 duration 求和')
  assert.ok(html.includes('width:100%'), '空闲完成态进度条应为 100%')
})

test('生成中：显示预计剩余时间且随 progress 计算', () => {
  useShotStore.setState({ isGenerating: true, progress: 50 })
  const html = expectRender(BottomBar)
  // 剩余 = (100-50)*1.2 = 60 秒 → 01:00
  assert.ok(html.includes('预计剩余 01:00'), '剩余时间应按 (100-progress)*1.2 秒折算')
})

test('生成中进度条宽度使用 progress 值', () => {
  useShotStore.setState({ isGenerating: true, progress: 40 })
  const html = expectRender(BottomBar)
  assert.ok(html.includes('width:40%'))
})

test('WebSocket 已连接显示实时，否则显示兜底轮询', () => {
  useTaskStore.setState({ connectionState: 'open' })
  assert.ok(expectRender(BottomBar).includes('实时'))

  useTaskStore.setState({ connectionState: 'reconnecting' })
  assert.ok(expectRender(BottomBar).includes('兜底轮询'))
})

test('后台任务摘要：失败任务数大于 0 时显示失败计数', () => {
  useTaskStore.setState({ summary: { ...emptySummary(), activeCount: 2, failedCount: 3 } })
  // React SSR 会在文本与表达式之间插入 <!-- --> 注释节点，断言前剥离。
  const html = expectRender(BottomBar).replace(/<!--.*?-->/g, '')
  assert.ok(html.includes('后台任务 2 进行中'), '应显示进行中数量')
  assert.ok(html.includes('3 失败'), '应显示失败数量')
})

test('当前步骤使用 progressStepLabel 映射', () => {
  useShotStore.setState({ currentStep: 'generate_storyboard' })
  const html = expectRender(BottomBar)
  assert.ok(html.includes('当前步骤：'), '应展示当前步骤')
})

test('无步骤时显示待机步骤标签', () => {
  const html = expectRender(BottomBar)
  assert.ok(html.includes('当前步骤：'))
  assert.ok(!html.includes('width:NaN'), '进度条不应出现 NaN')
})

test('调试面板入口按钮带可访问性属性', () => {
  const html = expectRender(BottomBar)
  assert.ok(html.includes('aria-label="打开任务调试日志"'), '折叠态 aria-label')
  assert.ok(html.includes('aria-label="打开后台任务中心"'))
})
