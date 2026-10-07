import assert from 'node:assert/strict'
import {
  beginProjectNavigationIntent,
  currentProjectNavigationIntent,
  registerProjectNavigationGuard,
  requestProjectNavigation,
} from './projectNavigationGuard.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  const beforeIntent = currentProjectNavigationIntent()
  assert.equal(beginProjectNavigationIntent(), beforeIntent + 1)
  assert.equal(currentProjectNavigationIntent(), beforeIntent + 1)

  let calls = 0
  const unregisterSuccess = registerProjectNavigationGuard(async (from, to) => {
    calls += 1
    return from === 'project-a' && to === 'project-b'
  })

  assert.equal(await requestProjectNavigation('project-a', 'project-b'), true)
  assert.equal(calls, 1, 'a registered guard should run before project navigation')
  assert.equal(await requestProjectNavigation('project-a', 'project-a'), true)
  assert.equal(calls, 1, 'navigating to the current project should not flush it')
  unregisterSuccess()

  const unregisterFailure = registerProjectNavigationGuard(async () => false)
  assert.equal(await requestProjectNavigation('project-a', 'project-b'), false, 'a failed save should block navigation')
  unregisterFailure()

  const unregisterRejected = registerProjectNavigationGuard(async () => {
    throw new Error('save failed')
  })
  assert.equal(await requestProjectNavigation('project-a', null), false, 'a rejected guard should fail closed')
  unregisterRejected()

  assert.equal(await requestProjectNavigation('project-a', 'project-b'), true, 'unregistered guards must not leak')
})
