import assert from 'node:assert/strict'
import { backendFailurePageUrl, renderBackendFailurePage } from './failurePage.ts'

const page = renderBackendFailurePage('后端健康检查超时', '可点击"重试启动"')

assert.match(page, /后端健康检查超时/, 'the detail message must be embedded')
assert.match(page, /重试启动/, 'the retry action must be offered')
assert.match(page, /<meta http-equiv="Content-Security-Policy"/, 'the page must pin its own CSP')

const injected = renderBackendFailurePage('<script>alert(1)</script>', '"><img src=x onerror=alert(1)>')
assert.doesNotMatch(injected, /<script>alert\(1\)<\/script>"/, 'raw script tags from the error detail must not survive')
assert.ok(!injected.includes('"><img src=x'), 'attribute injection via the hint must be escaped')
assert.ok(injected.includes('&lt;script&gt;'), 'the detail must be HTML-escaped')

const url = backendFailurePageUrl('x', 'y')
assert.match(url, /^data:text\/html;charset=utf-8,/, 'the failure page must load from a data URL')
assert.ok(!url.includes(' '), 'the URL must be percent-encoded')
console.log('failurePage tests passed')
