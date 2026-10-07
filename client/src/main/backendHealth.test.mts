import assert from 'node:assert/strict'
import { isComicAgentHealthResponse } from './backendHealth.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  assert.equal(
    isComicAgentHealthResponse('{"status":"ok","service":"comic-agent"}'),
    true,
    'the ComicAgent health marker should be accepted',
  )
  assert.equal(
    isComicAgentHealthResponse('{"status":"ok","service":"another-service"}'),
    false,
    'a different service on the same port must be rejected',
  )
  assert.equal(
    isComicAgentHealthResponse('{"status":"ok","service":"Comic-Agent"}'),
    false,
    'the service marker comparison must be case-sensitive',
  )
  assert.equal(
    isComicAgentHealthResponse('{"status":"ok"}'),
    false,
    'a response without the service marker must be rejected',
  )
  assert.equal(
    isComicAgentHealthResponse('{"status":"starting","service":"comic-agent"}'),
    false,
    'a backend that is not ready must be rejected',
  )
  assert.equal(isComicAgentHealthResponse('not json'), false, 'malformed JSON must be rejected')
  assert.equal(isComicAgentHealthResponse('null'), false, 'non-object JSON must be rejected')
})
