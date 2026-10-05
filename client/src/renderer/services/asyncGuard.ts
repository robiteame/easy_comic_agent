export interface ProjectAsyncSnapshot {
  projectId: string | null
  projectEpoch: number
  operationToken: number
}

export function isCurrentProjectAsyncSnapshot(
  expected: ProjectAsyncSnapshot,
  current: ProjectAsyncSnapshot,
  mounted = true,
): boolean {
  return (
    mounted &&
    expected.projectId === current.projectId &&
    expected.projectEpoch === current.projectEpoch &&
    expected.operationToken === current.operationToken
  )
}

export function isLatestResourceResponse(
  requestId: number,
  latestRequestId: number,
  mutationVersion: number,
  currentMutationVersion: number,
): boolean {
  return requestId === latestRequestId && mutationVersion === currentMutationVersion
}
