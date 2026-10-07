import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  buildConnectionTestPayload,
  connectionTestCapability,
  connectionTestFailureFromError,
  connectionTestText,
  connectionTestTone,
  normalizeConnectionTestResult,
} from './systemSettingsModel.ts'

test('connectionTestCapability 把表单类别映射为接口能力名', () => {
  assert.equal(connectionTestCapability('script'), 'llm')
  assert.equal(connectionTestCapability('image'), 'image')
  assert.equal(connectionTestCapability('video'), 'video')
  assert.equal(connectionTestCapability('voice'), 'tts')
  assert.equal(connectionTestCapability('unknown'), null)
  assert.equal(connectionTestCapability(''), null)
})

test('buildConnectionTestPayload 携带草稿表单值并剔除后端回显字段', () => {
  const payload = buildConnectionTestPayload('voice', {
    protocol: 'mimo-tts',
    base_url: 'https://example.test/v1',
    api_key: '********',
    model: 'mimo-tts-x',
    voice: '冰糖',
    capabilities: { reference_images: false },
  })
  assert.ok(payload)
  assert.equal(payload.capability, 'tts')
  // 掩码密钥原样上送，由服务端判断是否复用已保存密钥。
  assert.equal(payload.config.api_key, '********')
  assert.equal(payload.config.model, 'mimo-tts-x')
  assert.equal(payload.config.voice, '冰糖')
  assert.equal('capabilities' in payload.config, false)
})

test('buildConnectionTestPayload 对未知类别返回 null，空表单生成空配置', () => {
  assert.equal(buildConnectionTestPayload('pricing', { base_url: 'x' }), null)
  const emptyForm = buildConnectionTestPayload('script', null)
  assert.ok(emptyForm)
  assert.equal(emptyForm.capability, 'llm')
  assert.deepEqual(emptyForm.config, {})
})

test('normalizeConnectionTestResult 保留合法结果并钳制延迟', () => {
  const result = normalizeConnectionTestResult({
    status: 'ok',
    provider: 'openai-chat',
    model: 'm1',
    latency_ms: 812.4,
    message: 'ok',
  })
  assert.ok(result)
  assert.equal(result.status, 'ok')
  assert.equal(result.latency_ms, 812)

  const negative = normalizeConnectionTestResult({ status: 'fail', latency_ms: -5 })
  assert.ok(negative)
  assert.equal(negative.latency_ms, 0)
  assert.equal(negative.message, '')
})

test('normalizeConnectionTestResult 拒绝不合法结构', () => {
  assert.equal(normalizeConnectionTestResult(null), null)
  assert.equal(normalizeConnectionTestResult('ok'), null)
  assert.equal(normalizeConnectionTestResult({ status: 'pending' }), null)
  assert.equal(normalizeConnectionTestResult({ provider: 'openai-chat' }), null)
})

test('connectionTestTone 映射展示色调', () => {
  assert.equal(connectionTestTone('ok'), 'success')
  assert.equal(connectionTestTone('fail'), 'error')
  assert.equal(connectionTestTone('unsupported_check'), 'muted')
})

test('connectionTestText 成功带延迟，失败原样透传后端 message', () => {
  const ok = normalizeConnectionTestResult({ status: 'ok', latency_ms: 812, message: '补全请求成功' })
  assert.ok(ok)
  assert.equal(connectionTestText(ok), '连接成功 · 812ms')

  const instant = normalizeConnectionTestResult({ status: 'ok', latency_ms: 0, message: '' })
  assert.ok(instant)
  assert.equal(connectionTestText(instant), '连接成功')

  const fail = normalizeConnectionTestResult({
    status: 'fail',
    latency_ms: 120,
    message: '鉴权失败（HTTP 401）：API Key 无效或没有权限。',
  })
  assert.ok(fail)
  assert.equal(connectionTestText(fail), '鉴权失败（HTTP 401）：API Key 无效或没有权限。')
  assert.equal(
    connectionTestText({ status: 'fail', provider: '', model: '', latency_ms: 0, message: '' }),
    '连接失败，请检查配置',
  )

  const unsupported = normalizeConnectionTestResult({
    status: 'unsupported_check',
    latency_ms: 30,
    message: '该服务未提供模型列表接口。',
  })
  assert.ok(unsupported)
  assert.equal(connectionTestText(unsupported), '该服务未提供模型列表接口。')
})

test('connectionTestFailureFromError 优先展示服务端 detail', () => {
  const withDetail = connectionTestFailureFromError({
    response: { data: { detail: '不支持的能力类别' } },
    message: 'Request failed with status code 400',
  })
  assert.equal(withDetail.status, 'fail')
  assert.equal(withDetail.message, '不支持的能力类别')

  const plain = connectionTestFailureFromError(new Error('网络中断'))
  assert.equal(plain.message, '网络中断')

  const fallback = connectionTestFailureFromError('weird')
  assert.equal(fallback.message, '连接测试失败，请稍后重试')
})
