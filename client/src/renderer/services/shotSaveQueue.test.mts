import {
  drainPendingSaves,
  retirePendingSave,
  retirePendingSaveAfterFlush,
  type PendingSaveEntry,
} from './shotSaveQueue.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  function assert(condition: unknown, message: string) {
    if (!condition) throw new Error(message)
  }

  function deferred<T>() {
    let resolve!: (value: T) => void
    const promise = new Promise<T>((nextResolve) => {
      resolve = nextResolve
    })
    return { promise, resolve }
  }

  const entry: PendingSaveEntry<Record<string, unknown>> = {
    pending: { visual_notes: 'first draft' },
    inFlight: false,
    failed: false,
    promise: null,
    retired: false,
  }
  const firstSave = deferred<boolean>()
  const calls: Record<string, unknown>[] = []
  const flush = drainPendingSaves(entry, async (changes) => {
    calls.push(changes)
    if (calls.length === 1) return firstSave.promise
    return true
  })

  await Promise.resolve()
  entry.pending = { ...entry.pending, duration: 4 }
  firstSave.resolve(true)

  assert(await flush, 'explicit save should drain changes queued during an in-flight request')
  assert(calls.length === 2, 'queued changes should be saved in a follow-up request')
  assert(calls[0].visual_notes === 'first draft', 'first request should preserve its original snapshot')
  assert(calls[1].duration === 4, 'second request should include the in-flight edit')

  const failedEntry: PendingSaveEntry<Record<string, unknown>> = {
    pending: { emotion: 'tense' },
    inFlight: false,
    failed: false,
    promise: null,
    retired: false,
  }
  assert(!(await drainPendingSaves(failedEntry, async () => false)), 'a failed request should report failure')
  assert(failedEntry.pending.emotion === 'tense', 'a failed request should retain its changes for retry')

  const retiredEntry: PendingSaveEntry<Record<string, unknown>> = {
    pending: { dialogue: 'old project' },
    inFlight: false,
    failed: false,
    promise: null,
    retired: false,
  }
  const retiringSave = deferred<boolean>()
  let retiredCalls = 0
  const retiredFlush = drainPendingSaves(retiredEntry, async () => {
    retiredCalls += 1
    return retiringSave.promise
  })
  await Promise.resolve()
  retiredEntry.pending = { ...retiredEntry.pending, duration: 5 }
  retirePendingSave(retiredEntry)
  retiringSave.resolve(true)
  assert(!(await retiredFlush), 'a retired project queue should not report a successful flush')
  assert(retiredCalls === 1, 'a retired project queue should not submit edits accumulated in flight')

  const switchingEntry: PendingSaveEntry<Record<string, unknown>> = {
    pending: { dialogue: 'old project draft' },
    inFlight: false,
    failed: false,
    promise: null,
    retired: false,
  }
  const switchingSave = deferred<boolean>()
  const switchingCalls: Record<string, unknown>[] = []
  const switchingFlush = retirePendingSaveAfterFlush(switchingEntry, () =>
    drainPendingSaves(switchingEntry, async (changes) => {
      switchingCalls.push(changes)
      if (switchingCalls.length === 1) return switchingSave.promise
      return true
    }),
  )
  await Promise.resolve()
  assert(!switchingEntry.retired, 'a project switch must keep the old queue alive until its flush settles')
  switchingEntry.pending = { ...switchingEntry.pending, duration: 6 }
  switchingSave.resolve(true)
  assert(await switchingFlush, 'a project switch should report a successful queue drain')
  assert(switchingCalls.length === 2, 'a project switch should drain edits queued behind the active request')
  assert(switchingCalls[1].duration === 6, 'the follow-up project-switch save should preserve the latest edit')
  assert(switchingEntry.retired, 'the old project queue should retire after every pending edit is saved')

  const failedSwitchEntry: PendingSaveEntry<Record<string, unknown>> = {
    pending: { dialogue: 'must survive navigation' },
    inFlight: false,
    failed: false,
    promise: null,
    retired: false,
  }
  assert(
    !(await retirePendingSaveAfterFlush(failedSwitchEntry, () =>
      drainPendingSaves(failedSwitchEntry, async () => false),
    )),
    'a failed project-switch flush should block navigation',
  )
  assert(!failedSwitchEntry.retired, 'a failed project-switch flush must remain retryable')
  assert(
    failedSwitchEntry.pending.dialogue === 'must survive navigation',
    'a failed project-switch flush must preserve the unsaved payload',
  )
})
