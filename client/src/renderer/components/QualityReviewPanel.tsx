import React, { useCallback, useEffect, useState } from 'react'
import Button from 'antd/es/button'
import message from 'antd/es/message'
import Tooltip from 'antd/es/tooltip'
import { ExclamationCircleOutlined, ReloadOutlined, SafetyCertificateOutlined } from '@ant-design/icons'
import { qualityReviewApi } from '../services/api'
import type { Shot } from '../stores/shotStore'
import {
  dimensionStatusMeta,
  formatScore,
  gateLine,
  hasUndetected,
  historyEntries,
  latestByStage,
  stageLabel,
  verdictMeta,
  type QualityCapabilityResponse,
  type QualityDimension,
  type QualityReviewRow,
} from './qualityReviewModel'

interface QualityReviewPanelProps {
  shot: Shot | null
}

/**
 * 镜头质量审核面板：每项评分、问题、证据、修正建议与历史候选结果。
 *
 * 展示原则与后端一致：结构检查 ≠ 质量通过；未检测（unsupported）与降级
 * 放行必须显式标注，绝不显示为「通过」。审核由自动模式质量门禁或手动
 * 复审触发，本面板只读展示 + 提供即时复审入口。
 */
function QualityReviewPanel({ shot }: QualityReviewPanelProps) {
  const [reviews, setReviews] = useState<QualityReviewRow[]>([])
  const [capability, setCapability] = useState<QualityCapabilityResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [reloading, setReloading] = useState(false)

  const shotId = shot?.id || ''

  const load = useCallback(async (targetShotId: string, silent = false) => {
    if (!targetShotId) return
    if (!silent) setLoading(true)
    try {
      const [shotResult, capabilityResult] = await Promise.all([
        qualityReviewApi.shotReviews(targetShotId),
        qualityReviewApi.capability().catch(() => null),
      ])
      setReviews(shotResult.reviews || [])
      setCapability(capabilityResult)
    } catch {
      // 后端不可达时保持空态；面板不阻塞审核工作流。
      setReviews([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(shotId)
  }, [shotId, load])

  useEffect(() => {
    // 自动模式质量门禁每轮审核都会推送该事件；面板就地刷新当前镜头。
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ shot_id?: string }>).detail || {}
      if (!detail.shot_id || detail.shot_id === shotId) void load(shotId, true)
    }
    window.addEventListener('quality-review-updated', handler)
    return () => window.removeEventListener('quality-review-updated', handler)
  }, [shotId, load])

  const rerun = async (stage: 'storyboard' | 'video') => {
    if (!shot) return
    setReloading(true)
    try {
      await qualityReviewApi.rerun(shot.id, stage)
      message.success(`${stageLabel(stage)}质量复审完成`)
      await load(shot.id, true)
    } catch (error) {
      const response = (error as { response?: { data?: { detail?: string } } })?.response
      message.error(response?.data?.detail || '质量复审失败，请检查审核 Provider 配置')
    } finally {
      setReloading(false)
    }
  }

  if (!shot) return null

  const latest = latestByStage(reviews)
  const current: QualityReviewRow | null = latest.video || latest.storyboard
  const vlmCapability = capability?.storyboard?.vlm
  const vlmUnsupported = vlmCapability && !vlmCapability.supported ? vlmCapability : null

  return (
    <div className="quality-panel panel-enter" role="region" aria-label="镜头质量审核">
      <div className="quality-panel-header">
        <strong><SafetyCertificateOutlined /> 质量审核</strong>
        <span className="quality-panel-sub">
          {capability ? gateLine(capability.gate) : '门禁配置读取中…'}
        </span>
        <div className="quality-panel-actions">
          <Button
            size="small"
            icon={<ReloadOutlined />}
            loading={reloading}
            disabled={!shot.storyboard_path && !shot.image_path}
            onClick={() => void rerun('storyboard')}
          >
            复审故事板
          </Button>
          <Button
            size="small"
            icon={<ReloadOutlined />}
            loading={reloading}
            disabled={!shot.video_path}
            onClick={() => void rerun('video')}
          >
            复审视频
          </Button>
        </div>
      </div>

      {vlmUnsupported && (
        <div className="quality-capability-warning" role="alert">
          <ExclamationCircleOutlined />
          <span>
            审核能力未配置：{vlmUnsupported.reason || 'VLM 未配置'}。
            自动模式不会自动批准任何镜头（裁决为「未检测」），请配置 VLM 或转人工审核。
            {capability?.storyboard?.identity_embedding && !capability.storyboard.identity_embedding.supported
              ? ` 另：${capability.storyboard.identity_embedding.reason || '身份 embedding 未配置'}，身份相似度证据将按未检测处理。`
              : ''}
          </span>
        </div>
      )}

      {loading ? (
        <div className="quality-empty">正在读取审核记录…</div>
      ) : !current ? (
        <div className="quality-empty">
          尚无质量审核记录。自动模式会在结构检查通过后逐镜头评分；也可点右上角「复审」手动触发。
        </div>
      ) : (
        <>
          <div className={`quality-verdict ${verdictMeta(current.verdict).tone}`}>
            <span className="quality-verdict-badge">{verdictMeta(current.verdict).label}</span>
            <span className="quality-verdict-score">{formatScore(current.overall_score)} 分</span>
            <span className="quality-verdict-meta">
              {stageLabel(current.stage)} · 第 {current.attempt} 轮 · 候选版本 v{current.shot_version}
            </span>
          </div>

          {hasUndetected(current) && (
            <div className="quality-degraded" role="alert">
              <ExclamationCircleOutlined />
              <span>
                存在未检测维度（{current.unsupported_dimensions.join('、') || '部分维度'}
                ）{current.degraded ? '，本次按降级策略放行——未检测不代表通过' : '，按当前策略不得放行'}。
              </span>
            </div>
          )}

          <div className="quality-dimensions">
            {current.dimensions.map((dimension: QualityDimension) => (
              <div key={dimension.key} className={`quality-dim ${dimensionStatusMeta(dimension.status).tone}`}>
                <div className="quality-dim-head">
                  <span className="quality-dim-label">{dimension.label}</span>
                  <span className={`quality-dim-status ${dimensionStatusMeta(dimension.status).tone}`}>
                    {dimensionStatusMeta(dimension.status).label}
                  </span>
                  {dimension.status === 'scored' && (
                    <span className="quality-dim-score">
                      <span className="quality-dim-score-bar">
                        <span
                          className="quality-dim-score-fill"
                          style={{ width: `${Math.max(2, Math.round((dimension.score || 0) * 100))}%` }}
                        />
                      </span>
                      <em>{formatScore(dimension.score)}</em>
                    </span>
                  )}
                  {dimension.provider && dimension.provider !== '-' && (
                    <Tooltip title={`评分来源：${dimension.provider}`}>
                      <span className="quality-dim-provider">{dimension.provider.split(':')[0]}</span>
                    </Tooltip>
                  )}
                </div>
                {(dimension.issues.length > 0 || dimension.status !== 'scored') && (
                  <ul className="quality-dim-issues">
                    {dimension.issues.map((issue, index) => (
                      <li key={index}>{issue}</li>
                    ))}
                    {dimension.issues.length === 0 && dimension.status === 'skipped' && (
                      <li className="quality-dim-note">{Array.isArray(dimension.evidence?.reason) ? '' : String(dimension.evidence?.reason || '该镜头不适用')}</li>
                    )}
                    {dimension.issues.length === 0 && dimension.status === 'unsupported' && (
                      <li className="quality-dim-note">能力未配置，未参与评分（绝不计为通过）</li>
                    )}
                  </ul>
                )}
                {dimension.status === 'scored' &&
                  Array.isArray(dimension.evidence?.vlm) &&
                  (dimension.evidence.vlm as string[]).length > 0 && (
                    <div className="quality-dim-evidence">
                      依据：{(dimension.evidence.vlm as string[]).slice(0, 3).join('；')}
                    </div>
                  )}
                {Boolean(dimension.evidence && typeof dimension.evidence === 'object' && dimension.evidence.identity_embedding) && (
                  <div className="quality-dim-evidence">
                    身份相似度：
                    {(() => {
                      const embedding = dimension.evidence.identity_embedding as Record<string, unknown>
                      const items = Array.isArray(embedding.similarities)
                        ? (embedding.similarities as Array<{ label?: string; score?: number }>)
                        : []
                      return items.length
                        ? `${items.map((item) => `${item.label || '?'} ${formatScore(item.score ?? null)}分`).join('、')}（阈值 ${formatScore(Number(embedding.threshold) || null)} 分）`
                        : String((embedding as { reason?: string; status?: string }).reason || (embedding as { status?: string }).status || '未执行')
                    })()}
                  </div>
                )}
              </div>
            ))}
          </div>

          {(current.suggestion || (current.prompt_fix?.directives || []).length > 0) && (
            <div className="quality-suggestion">
              <strong>修正建议</strong>
              {current.suggestion && <p>{current.suggestion}</p>}
              {(current.prompt_fix?.directives || []).length > 0 && (
                <ul>
                  {current.prompt_fix!.directives!.map((directive, index) => (
                    <li key={index}>{directive}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div className="quality-history">
            <strong>历史候选</strong>
            {historyEntries(reviews).map((entry) => (
              <div key={entry.id} className={`quality-history-item ${verdictMeta(entry.verdict).tone}`}>
                <span>{entry.title}</span>
                <span className="quality-history-verdict">{verdictMeta(entry.verdict).label}</span>
                <span>{entry.score} 分</span>
                {entry.degraded && <span className="quality-history-degraded">降级放行</span>}
                <span className="quality-history-time">{entry.created_at ? entry.created_at.slice(5, 16).replace('T', ' ') : ''}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

export default QualityReviewPanel
