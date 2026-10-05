import assert from 'node:assert/strict'

import type { JobDebugEvent } from '../services/jobTypes.ts'
import {
  apiResultFor,
  debugFilterMatches,
  latestApiRequest,
  mergeDebugEvents,
  progressStepLabel,
  promptSections,
} from './progressDebugModel.ts'

function event(overrides: Partial<JobDebugEvent> = {}): JobDebugEvent {
  return {
    id: 'event-1',
    request_id: '',
    timestamp: '2026-09-30T12:00:00',
    kind: 'progress',
    level: 'progress',
    step: 'generate_storyboard',
    progress: 42,
    message: '正在生成分镜',
    api: '',
    provider: '',
    model: '',
    params: '',
    prompt: '',
    detail: '',
    ...overrides,
  }
}

assert.equal(progressStepLabel('generate_storyboard'), '分镜生成')
assert.equal(progressStepLabel('custom_stage'), 'custom stage')

const request = event({
  id: 'request',
  kind: 'api_request',
  level: 'request',
  request_id: 'call-1',
  api: 'LLM JSON',
  params: { temperature: 0.3 },
  prompt: { system: 'system', user: 'user' },
})
const result = event({
  id: 'result',
  kind: 'api_result',
  level: 'success',
  request_id: 'call-1',
  api: 'LLM JSON',
  message: 'LLM JSON 调用成功',
})
const progress = event({ id: 'progress' })

assert.equal(latestApiRequest([progress, request, result])?.id, 'request')
assert.equal(apiResultFor([request, result], request)?.id, 'result')
assert.equal(debugFilterMatches(request, 'api'), true)
assert.equal(debugFilterMatches(progress, 'api'), false)
assert.equal(debugFilterMatches(event({ level: 'error' }), 'error'), true)

const merged = mergeDebugEvents([result, progress], [request, request])
assert.deepEqual(
  merged.map((item) => item.id),
  ['progress', 'request', 'result'],
)

assert.deepEqual(promptSections({ system: 'A', user: 'B' }), [
  { label: '系统提示词', content: 'A' },
  { label: '用户提示词', content: 'B' },
])
assert.equal(promptSections('hello')[0].content, 'hello')
