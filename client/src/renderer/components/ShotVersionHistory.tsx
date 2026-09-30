import React, { useEffect, useMemo, useRef, useState } from 'react'
import Button from 'antd/es/button'
import Drawer from 'antd/es/drawer'
import message from 'antd/es/message'
import Modal from 'antd/es/modal'
import Tag from 'antd/es/tag'
import Tooltip from 'antd/es/tooltip'
import { HistoryOutlined, ReloadOutlined, UndoOutlined } from '@ant-design/icons'
import { describeShotVersionError, shotApi, toOutputUrl } from '../services/api'
import type {
  ShotVersionCompareResponse,
  ShotVersionSummary,
} from '../services/shotVersionTypes'
import type { Shot } from '../stores/shotStore'
import {
  canCompare,
  changedFieldCount,
  defaultCompareSelection,
  fieldLabel,
  formatSnapshotValue,
  formatVersionTime,
  mediaPreviewOf,
  nextCompareSelection,
  normalizeVersionList,
  restoreAvailability,
  restoreConfirmContent,
  sourceLabel,
  visibleDiffRows,
  type CompareSelection,
} from './shotVersionModel'

interface ShotVersionHistoryProps {
  open: boolean
  shot: Shot | null
  onClose: () => void
  /** 恢复成功后回调：把服务端返回的最新镜头状态合并回全局 store。 */
  onRestored: (shot: Record<string, unknown>, restoredFrom: ShotVersionSummary) => void
  /**
   * 恢复前钩子：父组件先冲掉防抖保存队列并等待落库，避免未保存的编辑在
   * 恢复之后又被保存请求写回。返回 false 表示保存失败，应中止恢复。
   */
  beforeRestore?: () => Promise<boolean>
}

/**
 * 镜头「版本历史」抽屉：时间线 + A/B 并排预览 + 字段差异 + 恢复。
 *
 * 对比与时间线全部只读；恢复走后端追加式接口，成功后刷新时间线并把最新
 * 镜头状态交回父组件合并。已审核锁定的镜头只能查看与对比，不能恢复。
 */
const ShotVersionHistory: React.FC<ShotVersionHistoryProps> = ({ open, shot, onClose, onRestored, beforeRestore }) => {
  const [versions, setVersions] = useState<ShotVersionSummary[]>([])
  const [currentVersionId, setCurrentVersionId] = useState<string | null>(null)
  const [listLoading, setListLoading] = useState(false)
  const [listError, setListError] = useState('')
  const [selection, setSelection] = useState<CompareSelection>({ a: null, b: null })
  const [compare, setCompare] = useState<ShotVersionCompareResponse | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareError, setCompareError] = useState('')
  const [showUnchanged, setShowUnchanged] = useState(false)
  const [restoringId, setRestoringId] = useState<string | null>(null)

  const mountedRef = useRef(false)
  const shotIdRef = useRef<string | null>(null)
  const listRequestRef = useRef(0)
  const compareRequestRef = useRef(0)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const isCurrentShotContext = (shotId: string) =>
    mountedRef.current && shotIdRef.current === shotId && open

  const loadVersions = async (shotId: string, keepSelection = false) => {
    const requestId = ++listRequestRef.current
    setListLoading(true)
    try {
      const response = await shotApi.versions(shotId)
      if (requestId !== listRequestRef.current || !isCurrentShotContext(shotId)) return
      const list = normalizeVersionList(response.versions)
      setVersions(list)
      setCurrentVersionId(response.current_version_id)
      setListError('')
      if (!keepSelection) setSelection(defaultCompareSelection(list))
    } catch (err) {
      if (requestId !== listRequestRef.current || !isCurrentShotContext(shotId)) return
      setVersions([])
      setCurrentVersionId(null)
      setListError(describeShotVersionError(err, '版本列表加载失败'))
    } finally {
      if (requestId === listRequestRef.current && mountedRef.current) setListLoading(false)
    }
  }

  useEffect(() => {
    const shotId = shot?.id || null
    shotIdRef.current = shotId
    if (!open || !shotId) {
      setVersions([])
      setCurrentVersionId(null)
      setSelection({ a: null, b: null })
      setCompare(null)
      setListError('')
      setCompareError('')
      return
    }
    void loadVersions(shotId)
  }, [open, shot?.id])

  useEffect(() => {
    if (!open || !canCompare(selection)) {
      setCompare(null)
      setCompareError('')
      return
    }
    const shotId = shotIdRef.current
    if (!shotId) return
    const aId = selection.a as string
    const bId = selection.b as string
    const requestId = ++compareRequestRef.current
    setCompareLoading(true)
    shotApi
      .compareVersions(shotId, aId, bId)
      .then((result) => {
        if (requestId !== compareRequestRef.current || !isCurrentShotContext(shotId)) return
        setCompare(result)
        setCompareError('')
      })
      .catch((err) => {
        if (requestId !== compareRequestRef.current || !isCurrentShotContext(shotId)) return
        setCompare(null)
        setCompareError(describeShotVersionError(err, '版本对比加载失败'))
      })
      .finally(() => {
        if (requestId === compareRequestRef.current && mountedRef.current) setCompareLoading(false)
      })
  }, [open, selection.a, selection.b])

  const pickSlot = (slot: 'a' | 'b', versionId: string) => {
    setSelection((current) => nextCompareSelection(current, slot, versionId))
  }

  const handleRestore = (version: ShotVersionSummary) => {
    const shotId = shotIdRef.current
    if (!shotId) return
    Modal.confirm({
      title: `恢复到版本 v${version.number || '?'}`,
      content: <div style={{ whiteSpace: 'pre-line' }}>{restoreConfirmContent(version)}</div>,
      okText: '确认恢复',
      cancelText: '取消',
      onOk: async () => {
        if (beforeRestore) {
          const saved = await beforeRestore()
          if (!saved) {
            message.warning('镜头修改保存失败，已取消恢复。请先点击“保存修改”后重试。')
            return
          }
        }
        setRestoringId(version.id)
        try {
          const result = await shotApi.restoreVersion(shotId, version.id)
          if (!isCurrentShotContext(shotId)) return
          message.success(`已恢复到 v${version.number}，并创建新版本记录`)
          onRestored(result.shot, version)
          await loadVersions(shotId, true)
        } catch (err) {
          message.error(describeShotVersionError(err, '恢复失败'))
        } finally {
          setRestoringId(null)
        }
      },
    })
  }

  const diffRows = useMemo(
    () => (compare ? visibleDiffRows(compare.diff, showUnchanged) : []),
    [compare, showUnchanged],
  )
  const changedCount = useMemo(() => (compare ? changedFieldCount(compare.diff) : 0), [compare])

  const renderMediaPane = (side: 'A' | 'B') => {
    const detail = side === 'A' ? compare?.a : compare?.b
    if (!detail) return null
    const media = mediaPreviewOf(detail.snapshot)
    const imageUrl = toOutputUrl(media.image)
    const videoUrl = toOutputUrl(media.video)
    return (
      <div className="version-media-pane">
        <div className="version-media-head">
          <strong>{side} · v{detail.number}</strong>
          <span>{sourceLabel(detail.source)}</span>
        </div>
        <div className="version-media-body">
          {videoUrl ? (
            <video src={videoUrl} controls preload="metadata" poster={imageUrl || undefined} />
          ) : imageUrl ? (
            <img src={imageUrl} alt={`版本 v${detail.number} 预览`} loading="lazy" decoding="async" />
          ) : (
            <span className="version-media-empty">该版本没有可预览的媒体</span>
          )}
        </div>
        <div className="version-media-meta">
          <span>{formatVersionTime(detail.created_at)}</span>
          {detail.task_id && <code>{detail.task_id}</code>}
        </div>
      </div>
    )
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      width="min(760px, 96vw)"
      title={
        <span className="version-drawer-title">
          <HistoryOutlined /> 版本历史
          {shot ? ` · 镜头 ${shot.sequence || 1}` : ''}
        </span>
      }
      extra={
        <Button
          size="small"
          icon={<ReloadOutlined />}
          loading={listLoading}
          disabled={!shot}
          onClick={() => shot && void loadVersions(shot.id, true)}
        >
          刷新
        </Button>
      }
      className="shot-version-drawer"
      destroyOnClose
    >
      {!shot && <div className="version-empty-hint">选择镜头后可查看版本历史。</div>}

      {shot && (
        <div className="version-history-body">
          {shot.confirmed && (
            <div className="locked-shot-note">该镜头已审核锁定：可查看与对比版本，但不能恢复。</div>
          )}

          {listError && <div className="version-list-error" role="alert">{listError}</div>}

          {listLoading && !versions.length && <div className="version-empty-hint">正在加载版本记录…</div>}

          {!listLoading && !listError && !versions.length && (
            <div className="version-empty-hint">该镜头还没有版本记录。编辑、重新生成或导入都会自动保存版本。</div>
          )}

          {versions.length > 0 && (
            <>
              <div className="version-timeline-hint">
                点击 <b>A</b> / <b>B</b> 选择两个版本进行对比；恢复会创建新版本，不会改动历史。
              </div>
              <ol className="version-timeline" aria-label="版本时间线">
                {versions.map((version) => {
                  const restoreState = restoreAvailability(version, {
                    confirmed: Boolean(shot.confirmed),
                    currentVersionId,
                  })
                  const isA = selection.a === version.id
                  const isB = selection.b === version.id
                  return (
                    <li
                      key={version.id}
                      className={`version-item${isA ? ' picked-a' : ''}${isB ? ' picked-b' : ''}${currentVersionId === version.id ? ' is-current' : ''}`}
                    >
                      <div className="version-item-dot" aria-hidden="true" />
                      <div className="version-item-main">
                        <div className="version-item-head">
                          <strong>v{version.number}</strong>
                          <Tag className={`version-source-tag ${version.source}`}>{sourceLabel(version.source)}</Tag>
                          {currentVersionId === version.id && <Tag color="green">当前状态</Tag>}
                          <span className="version-item-time">{formatVersionTime(version.created_at)}</span>
                        </div>
                        <div className="version-item-meta">
                          <span>镜头版本号 {version.version}</span>
                          {version.has_image && <span>有故事板</span>}
                          {version.has_video && <span>有视频</span>}
                          {version.task_id && <code>{version.task_id}</code>}
                        </div>
                      </div>
                      <div className="version-item-actions">
                        <button
                          type="button"
                          className={`version-pick slot-a${isA ? ' active' : ''}`}
                          aria-pressed={isA}
                          onClick={() => pickSlot('a', version.id)}
                        >
                          A
                        </button>
                        <button
                          type="button"
                          className={`version-pick slot-b${isB ? ' active' : ''}`}
                          aria-pressed={isB}
                          onClick={() => pickSlot('b', version.id)}
                        >
                          B
                        </button>
                        <Tooltip title={restoreState.allowed ? `恢复到 v${version.number}` : restoreState.reason}>
                          <Button
                            size="small"
                            icon={<UndoOutlined />}
                            loading={restoringId === version.id}
                            disabled={!restoreState.allowed}
                            onClick={() => handleRestore(version)}
                          >
                            恢复
                          </Button>
                        </Tooltip>
                      </div>
                    </li>
                  )
                })}
              </ol>
            </>
          )}

          {(selection.a || selection.b) && (
            <section className="version-compare" aria-label="A/B 版本对比">
              <div className="version-compare-head">
                <h4>
                  A/B 对比
                  {compare && <em>{changedCount} 个字段有变化</em>}
                </h4>
                {compare && (
                  <button type="button" className="version-diff-toggle" onClick={() => setShowUnchanged((v) => !v)}>
                    {showUnchanged ? '只看变化' : '显示全部字段'}
                  </button>
                )}
              </div>

              {!canCompare(selection) && (
                <div className="version-empty-hint">再选择一个不同的版本即可开始对比。</div>
              )}

              {canCompare(selection) && compareLoading && <div className="version-empty-hint">正在对比两个版本…</div>}
              {canCompare(selection) && !compareLoading && compareError && (
                <div className="version-list-error" role="alert">{compareError}</div>
              )}

              {compare && !compareLoading && (
                <>
                  <div className="version-media-grid">
                    {renderMediaPane('A')}
                    {renderMediaPane('B')}
                  </div>
                  <table className="version-diff-table">
                    <thead>
                      <tr>
                        <th>字段</th>
                        <th>A · v{compare.a.number}</th>
                        <th>B · v{compare.b.number}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diffRows.map((row) => (
                        <tr key={row.field} className={row.changed ? 'changed' : ''}>
                          <th scope="row">{fieldLabel(row.field)}</th>
                          <td>{formatSnapshotValue(row.a, row.field)}</td>
                          <td>{formatSnapshotValue(row.b, row.field)}</td>
                        </tr>
                      ))}
                      {!diffRows.length && (
                        <tr>
                          <td colSpan={3} className="version-diff-same">两个版本内容完全一致</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </>
              )}
            </section>
          )}
        </div>
      )}
    </Drawer>
  )
}

export default ShotVersionHistory
