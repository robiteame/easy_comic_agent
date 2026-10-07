import assert from 'node:assert/strict'
import { test } from 'vitest'

import { expectRender, visibleText } from '../test-support/ssrTestHelper.mts'
import { makeShot } from '../test-support/fixtures.mts'
import QualityReviewPanel from './QualityReviewPanel'

test('未选中镜头时不渲染面板', () => {
  const html = expectRender(QualityReviewPanel, { shot: null })
  assert.equal(html, '', 'shot=null 应返回空')
})

test('选中镜头时渲染质量审核区域与门禁配置读取中提示', () => {
  const html = visibleText(expectRender(QualityReviewPanel, { shot: makeShot() }))
  assert.ok(html.includes('质量审核'), '应显示面板标题')
  assert.ok(html.includes('门禁配置读取中'), 'SSR 下能力数据未加载，应显示读取中文案')
})

test('有故事板素材时复审故事板按钮可点击', () => {
  const html = expectRender(QualityReviewPanel, { shot: makeShot() })
  assert.ok(html.includes('复审故事板'))
  assert.ok(html.includes('复审视频'))
})

test('无视频素材时复审视频按钮禁用', () => {
  const html = expectRender(QualityReviewPanel, {
    shot: makeShot({ video_path: '', storyboard_path: '/s.png' }),
  })
  assert.ok(/复审视频/.test(html) && /disabled/.test(html), '无视频应禁用复审按钮')
})
