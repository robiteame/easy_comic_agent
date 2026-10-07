import assert from 'node:assert/strict'

import { describeJobApiError, jobApi } from './api.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  // --- 错误响应归一化（无权限 / 404 / 409 / 网络失败） ------------------------

  const forbidden = describeJobApiError({ response: { status: 403, data: {} } })
  assert.equal(forbidden.ok, false)
  assert.equal(forbidden.status, 'forbidden')
  assert.match(forbidden.message, /HTTP 403/, '无权限必须给出可读提示')

  const notFound = describeJobApiError({
    response: { status: 404, data: { message: '任务不存在或已被清理', error_code: 'job_not_found' } },
  })
  assert.equal(notFound.ok, false)
  assert.equal(notFound.status, 'not_found')
  assert.equal(notFound.message, '任务不存在或已被清理', '服务端的可读消息应原样透出')
  assert.equal(notFound.error_code, 'job_not_found')

  const conflict = describeJobApiError({
    response: {
      status: 409,
      data: {
        ok: false,
        status: 'scope_conflict',
        message: '同一项目下已有任务在运行',
        error_code: 'scope_conflict',
        job: null,
      },
    },
  })
  assert.equal(conflict.status, 'scope_conflict')
  assert.equal(conflict.message, '同一项目下已有任务在运行')

  const network = describeJobApiError(new Error('Network Error'))
  assert.equal(network.ok, false)
  assert.equal(network.status, 'error')
  assert.match(network.message, /网络不可用/, '网络失败不能静默吞掉')

  const unknownShape = describeJobApiError({ response: { status: 500, data: 'boom' } })
  assert.match(unknownShape.message, /HTTP 500/)

  // --- 请求封装：参数与路径 --------------------------------------------------

  const calls: { method: string; url: string; params?: unknown }[] = []
  const api = (await import('./api.ts')).default

  api.defaults.adapter = (async (config: { method?: string; url?: string; params?: unknown }) => {
    const method = String(config.method || 'get').toLowerCase()
    calls.push({ method, url: String(config.url), params: config.params })
    return {
      data:
        method === 'get'
          ? {
              items: [],
              total: 0,
              page: 1,
              page_size: 100,
              pages: 1,
              active_count: 0,
              status_counts: {},
              job_type_counts: {},
              generated_at: '',
            }
          : { ok: true, status: 'cancelled', message: '已请求取消' },
      status: 200,
      statusText: 'OK',
      headers: {},
      config,
    }
  }) as never

  await jobApi.list({ project_id: 'p1', status: ['failed'], page: 2, page_size: 20 })
  assert.equal(calls[0].method, 'get')
  assert.equal(calls[0].url, '/api/jobs')
  assert.deepEqual(calls[0].params, { project_id: 'p1', status: ['failed'], page: 2, page_size: 20 })

  await jobApi.cancel('job id/../x')
  assert.equal(calls[1].method, 'post')
  assert.equal(calls[1].url, '/api/jobs/job%20id%2F..%2Fx/cancel', '任务 ID 必须转义后再拼进路径')

  await jobApi.retry('job-1')
  assert.equal(calls[2].url, '/api/jobs/job-1/retry')
  await jobApi.resume('job-1')
  assert.equal(calls[3].url, '/api/jobs/job-1/resume')
  await jobApi.remove('job-1')
  assert.equal(calls[4].method, 'delete')
  assert.equal(calls[4].url, '/api/jobs/job-1')
  await jobApi.cleanup({ project_id: 'p1' })
  assert.equal(calls[5].method, 'delete')
  assert.equal(calls[5].url, '/api/jobs')

  console.log('jobApi.test.mts ok')
})
