/**
 * 任务中心的前后端共享类型。
 *
 * 与后端 `server/services/job_dto.py` 的 DTO 一一对应：前端只依赖这些稳定字段，
 * 不解析中文提示、不假设内部实现。DTO 中**不含** run token，后端的错误消息也已
 * 脱敏截断。
 */

import type { JobCostDto, JobStatsCostDto } from './costTypes'

export type JobStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

export type JobType =
  | 'script_pipeline'
  | 'storyboard'
  | 'asset_generation'
  | 'shot_image'
  | 'shot_audio'
  | 'shot_video'
  | 'render'
  | 'unknown'

/**
 * 失败原因分析结果（后端规则 + LLM 异步生成）。
 *
 * `source` 为 "llm" 时 summary/suggestion 由模型生成；分析未完成或被跳过时
 * 整个字段为 null，界面回退到 error_code_label + error_message。
 */
export interface JobErrorDetail {
  summary: string
  suggestion: string
  source: string
  model: string
}

export interface JobDto {
  id: string
  scope: string
  project_id: string
  job_type: JobType
  job_type_label: string
  display_name: string
  status: JobStatus
  status_label: string
  progress: number
  current_step: string
  message: string
  error_code: string
  /** 失败类别的中文标签（服务端 error_code_label；历史行可能为空）。 */
  error_code_label: string
  error_message: string
  error_detail: JobErrorDetail | null
  attempt: number
  retry_of: string | null
  version: number
  created_at: string | null
  started_at: string | null
  updated_at: string | null
  finished_at: string | null
  cancel_requested_at: string | null
  /** 单次任务的实际耗时（墙钟秒数）。 */
  duration_seconds: number
  /**
   * 单次任务的成本与启动前估算。
   *
   * `cost_known=false` 时 `cost_micro` 必为 null，界面显示「成本未知」而不是 0。
   */
  cost: JobCostDto
  eta_seconds: number | null
  is_active: boolean
  is_terminal: boolean
  has_active_successor: boolean
  can_cancel: boolean
  can_retry: boolean
  can_resume: boolean
  can_delete: boolean
  retry_blocked_reason: string
  resume_blocked_reason: string
  batch_id?: string | null
  queue_position?: number
  priority?: number
  queue_order?: number
  stage?: string
  shot_id?: string | null
  dependency_ids?: string[]
  blocked_reason?: string
  queue_concurrency?: number
  paused?: boolean
  resume_missing?: boolean
  reuse_audio?: boolean
  force_confirmed?: boolean
  requested_version?: number
}

export interface JobAttemptDto {
  id: string
  attempt: number
  status: JobStatus
  error_code: string
  error_code_label: string
  error_message: string
  started_at: string | null
  finished_at: string | null
  updated_at: string | null
  duration_seconds: number
  retry_of: string | null
}

export interface JobDetailDto extends JobDto {
  generation_report?: Record<string, any>
  provider_capabilities?: Record<string, any>
  references_validated?: number | boolean
  references_sent?: unknown
  control_types_sent?: string[]
  reference_weight_policy?: string
  consistency_metrics?: Record<string, any>
  attempts: JobAttemptDto[]
  latest_attempt_job_id: string | null
  retry_relationship: {
    retry_of: string | null
    retry_of_attempt: number | null
    attempt: number
  }
}

export interface JobListResponse {
  items: JobDto[]
  total: number
  page: number
  page_size: number
  pages: number
  active_count: number
  status_counts: Record<string, number>
  job_type_counts: Record<string, number>
  generated_at: string
}

export interface JobStatsResponse {
  project_id?: string
  active_count: number
  failed_count: number
  total: number
  status_counts: Record<string, number>
  latest_job: JobDto | null
  /** 项目累计成本与调用次数（未知成本的调用单独计数）。 */
  cost: JobStatsCostDto
  generated_at: string
}

export interface JobActionResult {
  ok: boolean
  status: string
  message: string
  idempotent?: boolean
  error_code?: string
  job?: JobDto | null
}

export type JobEventType =
  | 'job.created'
  | 'job.updated'
  | 'job.progress'
  | 'job.completed'
  | 'job.failed'
  | 'job.cancelled'
  | 'job.interrupted'
  | 'job.retry_started'

export interface JobEvent {
  type: JobEventType | 'job_snapshot' | 'pong'
  job?: JobDto
  job_id?: string
  project_id?: string
  sent_at?: string
  jobs?: JobDto[]
  active_count?: number
  status_counts?: Record<string, number>
  total?: number
  page?: number
  page_size?: number
  generated_at?: string
}

export interface JobQueryParams {
  project_id?: string
  status?: JobStatus[]
  job_type?: JobType[]
  scope?: string
  active_only?: boolean
  q?: string
  page?: number
  page_size?: number
}
