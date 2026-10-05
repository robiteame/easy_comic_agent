import assert from 'node:assert/strict'
import { isCurrentProjectAsyncSnapshot, isLatestResourceResponse } from './asyncGuard.ts'

const request = { projectId: 'project-a', projectEpoch: 4, operationToken: 2 }
assert.equal(isCurrentProjectAsyncSnapshot(request, request), true)
assert.equal(
  isCurrentProjectAsyncSnapshot(request, { ...request, projectId: 'project-b' }),
  false,
  'a result from another project must be ignored',
)
assert.equal(
  isCurrentProjectAsyncSnapshot(request, { ...request, projectEpoch: 6 }),
  false,
  'switching away and back to the same project ID must still invalidate an old result',
)
assert.equal(
  isCurrentProjectAsyncSnapshot(request, { ...request, operationToken: 3 }),
  false,
  'a newer operation must supersede an older result in the same project epoch',
)
assert.equal(isCurrentProjectAsyncSnapshot(request, request, false), false, 'an unmounted consumer must reject results')

assert.equal(isLatestResourceResponse(5, 5, 8, 8), true)
assert.equal(isLatestResourceResponse(4, 5, 8, 8), false, 'an older GET response must not win')
assert.equal(
  isLatestResourceResponse(5, 5, 8, 9),
  false,
  'a GET response captured before a local mutation must not overwrite it',
)
assert.equal(
  isLatestResourceResponse(6, 6, 8, 8),
  true,
  'accepting an earlier read must not change the local mutation version for a newer read',
)
