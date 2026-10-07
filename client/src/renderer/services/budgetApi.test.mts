import assert from 'node:assert/strict'

import { budgetApi } from './api.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  // --- 请求封装：路径与参数 ------------------------------------------------

  const calls: { method: string; url: string; params?: unknown; data?: unknown }[] = []
  const api = (await import('./api.ts')).default

  api.defaults.adapter = (async (config: { method?: string; url?: string; params?: unknown; data?: unknown }) => {
    const method = String(config.method || 'get').toLowerCase()
    calls.push({ method, url: String(config.url), params: config.params, data: config.data })
    return { data: {}, status: 200, statusText: 'OK', headers: {}, config }
  }) as never

  await budgetApi.pricing()
  assert.equal(calls[0].method, 'get')
  assert.equal(calls[0].url, '/api/budget/pricing')

  await budgetApi.savePricing({
    currency: 'CNY',
    items: [
      {
        capability: 'image',
        provider: 'ark-seedream',
        model: '',
        unit_price_micro: 70000,
        unit_price_secondary_micro: null,
        resolution_multipliers: {},
        configured: true,
        note: '',
      },
    ],
  })
  assert.equal(calls[1].method, 'put')
  assert.equal(calls[1].url, '/api/budget/pricing')
  // axios 在进入 adapter 之前已经把请求体序列化成 JSON 字符串。
  const pricingBody = JSON.parse(String(calls[1].data)) as { currency: string; items: { unit_price_micro: number }[] }
  assert.equal(pricingBody.items[0].unit_price_micro, 70000, '单价必须以整数 micro 落库')

  await budgetApi.config('p1')
  assert.equal(calls[2].url, '/api/budget/config')
  assert.deepEqual(calls[2].params, { project_id: 'p1' })
  await budgetApi.config()
  assert.equal(calls[3].params, undefined, '不带项目时查询全局配置')

  await budgetApi.saveConfig({
    scope_type: 'project',
    scope_id: 'p1',
    soft_cost_micro: 50000000,
    hard_cost_micro: 80000000,
    soft_seconds: null,
    hard_seconds: null,
    enabled: true,
    note: '',
  })
  assert.equal(calls[4].method, 'put')
  assert.equal(calls[4].url, '/api/budget/config')

  await budgetApi.status('p1')
  assert.equal(calls[5].url, '/api/budget/status')
  assert.deepEqual(calls[5].params, { project_id: 'p1' })

  await budgetApi.estimate({ job_type: 'storyboard', project_id: 'p1', shot_ids: ['s1', 's2'] })
  assert.equal(calls[6].method, 'post')
  assert.equal(calls[6].url, '/api/budget/estimate')
  assert.deepEqual(JSON.parse(String(calls[6].data)), {
    job_type: 'storyboard',
    project_id: 'p1',
    shot_ids: ['s1', 's2'],
  })

  await budgetApi.summary({ project_id: 'p1' })
  assert.equal(calls[7].url, '/api/budget/summary')
  assert.deepEqual(calls[7].params, { project_id: 'p1' })
  await budgetApi.summary({ series_id: 'p0' })
  assert.deepEqual(calls[8].params, { series_id: 'p0' }, '剧集页用 series_id 汇总')

  // group_by 必须是重复查询参数，不能序列化成 group_by[]=（后端认不出这个参数名）
  await budgetApi.usage({
    project_id: 'p1',
    capability: 'video',
    group_by: ['job_type', 'shot'],
    page: 2,
    page_size: 20,
  })
  assert.equal(calls[9].method, 'get')
  assert.equal(
    calls[9].url,
    '/api/budget/usage?project_id=p1&capability=video&group_by=job_type&group_by=shot&page=2&page_size=20',
  )
  await budgetApi.usage({})
  assert.equal(calls[10].url, '/api/budget/usage', '没有筛选条件时不带问号')
  await budgetApi.usage({ group_by: [] })
  assert.equal(calls[11].url, '/api/budget/usage')

  await budgetApi.jobCost('job id/../x')
  assert.equal(calls[12].url, '/api/budget/jobs/job%20id%2F..%2Fx', '任务 ID 必须转义后再拼进路径')

  console.log('budgetApi.test.mts ok')
})
