/**
 * 镜头版本历史的 DTO 类型。
 *
 * 与后端 `services/shot_version_service.py` 的序列化字段一一对应；快照本体是
 * 「当时镜头字段的完整字典」，字段集合随版本演进，因此用宽松的 value 类型。
 */

export type ShotVersionSource = 'manual_edit' | 'regenerate' | 'restore' | 'import' | 'quality_retry'

export type ShotVersionSnapshot = Record<string, unknown>

export interface ShotVersionSummary {
  id: string
  shot_id: string
  number: number
  version: number
  source: ShotVersionSource
  task_id: string
  parent_version_id: string
  content_hash: string
  created_at: string | null
  has_image: boolean
  has_video: boolean
}

export interface ShotVersionDetail extends ShotVersionSummary {
  snapshot: ShotVersionSnapshot
}

export interface ShotVersionDiffRow {
  field: string
  a: unknown
  b: unknown
  changed: boolean
}

export interface ShotVersionListResponse {
  shot_id: string
  versions: ShotVersionSummary[]
  current_version_id: string | null
}

export interface ShotVersionCompareResponse {
  shot_id: string
  a: ShotVersionDetail
  b: ShotVersionDetail
  diff: ShotVersionDiffRow[]
  changed_fields: string[]
}

export interface ShotVersionRestoreResponse {
  id: string
  status: string
  version: number
  restored_from_version_id: string
  new_version_id: string | null
  shot: Record<string, unknown>
}
