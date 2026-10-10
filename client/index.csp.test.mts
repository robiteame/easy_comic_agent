import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'vitest'

test('CSP 允许 Electron 随机后端端口而不放开局域网', () => {
  const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8')
  const csp = html.match(/http-equiv="Content-Security-Policy"\s+content="([^"]+)"/)?.[1] || ''

  assert.match(csp, /connect-src[^;]*http:\/\/127\.0\.0\.1:\*/)
  assert.match(csp, /connect-src[^;]*ws:\/\/127\.0\.0\.1:\*/)
  assert.doesNotMatch(csp, /connect-src[^;]*8011/)
  assert.doesNotMatch(csp, /connect-src[^;]*https?:\/\/\*/)
})
