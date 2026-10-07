import type React from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ExperimentOutlined, FileImageOutlined, PlayCircleOutlined } from '@ant-design/icons'
import Button from 'antd/es/button'
import Input from 'antd/es/input'
import message from 'antd/es/message'
import { projectApi, settingsApi } from '../services/api'
import type { ModelConnectionTestResult } from '../services/api'
import { STYLE_DESCRIPTIONS, STYLE_OPTIONS } from '../constants/styleTemplates'
import {
  WIZARD_KEY_NOTE,
  WIZARD_STEPS,
  buildWizardModelConfigPatch,
  canGoNextFromStep,
  describeConnectionTest,
  nextWizardStep,
  prevWizardStep,
  wizardTestConfig,
  type WizardKeyForm,
  type WizardStepIndex,
} from './firstRunWizardModel'

/** 设置页「重新运行向导」通过该事件请求 App 打开向导（无需重置完成标志）。 */
export const OPEN_WIZARD_EVENT = 'workspace:open-wizard'

interface FirstRunWizardProps {
  open: boolean
  onClose: () => void
}

type TestFeedback = { tone: 'success' | 'error' | 'warning'; text: string }

/**
 * 首次启动三步向导：选画风 → 配置模型 → 开始创作。
 *
 * 数据全部走现有通道：Key 经 PUT /api/settings/model-configs 保存，示例项目
 * 经 POST /api/project/sample 创建，完成标志经 PUT /api/settings/onboarding
 * 持久化；向导本身不引入任何新存储。每一步都可以「上一步 / 跳过」。
 */
const FirstRunWizard: React.FC<FirstRunWizardProps> = ({ open, onClose }) => {
  const [step, setStep] = useState<WizardStepIndex>(0)
  const [selectedStyle, setSelectedStyle] = useState<string>('anime')
  const [keys, setKeys] = useState<WizardKeyForm>({ mimoApiKey: '', arkApiKey: '' })
  const [modelCategories, setModelCategories] = useState<Record<string, any> | null>(null)
  const [testing, setTesting] = useState<'llm' | 'video' | null>(null)
  const [testFeedback, setTestFeedback] = useState<Partial<Record<'llm' | 'video', TestFeedback>>>({})
  const [finishing, setFinishing] = useState<'sample' | 'blank' | null>(null)
  const keysSavedRef = useRef(false)

  // 打开向导时读取现行模型配置：Key 补丁需要判断端点是否已配置过，
  // 测试按钮也需要它来构造「保存前即可测」的待测配置。
  useEffect(() => {
    if (!open) return
    setStep(0)
    setTestFeedback({})
    keysSavedRef.current = false
    let active = true
    settingsApi
      .modelConfigs()
      .then((result: any) => {
        if (active) setModelCategories(result?.categories || {})
      })
      .catch(() => {
        // 读取失败不阻塞向导：保存时会按「未配置」路径使用内置预设。
        if (active) setModelCategories({})
      })
    return () => {
      active = false
    }
  }, [open])

  const updateKey = (field: keyof WizardKeyForm, value: string) => {
    setKeys((current) => ({ ...current, [field]: value }))
    keysSavedRef.current = false
    setTestFeedback((current) => ({ ...current, [field === 'mimoApiKey' ? 'llm' : 'video']: undefined }))
  }

  /** 经现有模型配置通道保存向导收集的 Key；未输入新 Key 时是空操作。 */
  const ensureKeysSaved = useCallback(async () => {
    if (keysSavedRef.current) return
    const patch = buildWizardModelConfigPatch(modelCategories, keys)
    if (!Object.keys(patch).length) return
    try {
      await settingsApi.saveModelConfigs({ categories: patch })
      keysSavedRef.current = true
    } catch (err: any) {
      // Key 是可选增强：保存失败不应阻断向导，如实提示后可稍后在设置页重试。
      message.error('密钥保存失败：' + (err?.response?.data?.detail || err?.message || '请稍后在系统设置中重试'))
    }
  }, [modelCategories, keys])

  const handleNext = async () => {
    const next = nextWizardStep(step)
    if (next == null) return
    if (!canGoNextFromStep(step, { style: selectedStyle, keys })) return
    if (step === 1) await ensureKeysSaved()
    setTestFeedback({})
    setStep(next)
  }

  const handlePrev = () => {
    const prev = prevWizardStep(step)
    if (prev != null) setStep(prev)
  }

  const finish = async (mode: 'skip' | 'sample' | 'blank') => {
    setFinishing(mode === 'skip' ? 'blank' : mode)
    try {
      if (mode !== 'skip') await ensureKeysSaved()

      if (mode === 'sample') {
        const created = await projectApi.createSample()
        const episode = created?.first_episode || created
        window.dispatchEvent(new CustomEvent('workspace:open-project', { detail: { projectId: episode.id } }))
        // 打开后直达故事板页：分镜列表 + 占位图首屏可见（项目切换会先
        // 重置回剧本页，这里延后切换以落在重置之后）。
        window.setTimeout(() => {
          window.dispatchEvent(
            new CustomEvent('workspace:navigate', { detail: { tab: 'storyboard', previewMode: 'shot' } }),
          )
        }, 600)
        message.success('示例项目已创建，可零 Key 浏览分镜与故事板')
      } else if (mode === 'blank') {
        const created = await projectApi.create({
          title: '未命名项目',
          first_episode_title: '第 1 集',
          style: selectedStyle,
          genre: '',
          output_format: '9:16',
          resolution: '1080p',
          platform: 'douyin',
        })
        const episode = created?.first_episode || created
        window.dispatchEvent(new CustomEvent('workspace:open-project', { detail: { projectId: episode.id } }))
        message.success(`已用「${styleLabel(selectedStyle)}」创建新项目，粘贴剧本即可开始`)
      }

      // 无论跳过还是完成，都记录「向导已运行」，第二次启动不再出现。
      try {
        await settingsApi.completeOnboarding(true)
      } catch {
        message.warning('向导完成状态保存失败，下次启动可能再次出现')
      }
      onClose()
    } catch (err: any) {
      message.error(
        mode === 'sample'
          ? '示例项目创建失败：' + (err?.response?.data?.detail || err?.message || '未知错误')
          : '新建项目失败：' + (err?.response?.data?.detail || err?.message || '未知错误'),
      )
    } finally {
      setFinishing(null)
    }
  }

  const handleTest = async (capability: 'llm' | 'video') => {
    setTesting(capability)
    try {
      const config = wizardTestConfig(capability, modelCategories, keys)
      const result = (await settingsApi.testModelConnection({ capability, config })) as ModelConnectionTestResult
      setTestFeedback((current) => ({ ...current, [capability]: describeConnectionTest(result) }))
    } catch (err: any) {
      setTestFeedback((current) => ({
        ...current,
        [capability]: { tone: 'error' as const, text: '测试请求失败：' + (err?.message || '网络不可用') },
      }))
    } finally {
      setTesting(null)
    }
  }

  if (!open) return null

  const canNext = canGoNextFromStep(step, { style: selectedStyle, keys })
  const prevStep = prevWizardStep(step)

  return (
    <div className="wizard-backdrop" role="dialog" aria-modal="true" aria-label="首次启动向导">
      <div className="wizard-panel">
        <header className="wizard-head">
          <div className="wizard-head-text">
            <div className="wizard-title">欢迎使用 ComicAgent 漫剧工作台</div>
            <div className="wizard-subtitle">三步开始：选一套画风，配好关键模型，然后打开示例看看能做什么。</div>
          </div>
          <ol className="wizard-step-indicator" aria-label="向导进度">
            {WIZARD_STEPS.map((item) => (
              <li
                key={item.key}
                className={item.key === step ? 'active' : item.key < step ? 'done' : ''}
                aria-current={item.key === step ? 'step' : undefined}
              >
                <span className="wizard-step-dot">{item.key < step ? '✓' : item.key + 1}</span>
                <span className="wizard-step-name">{item.title}</span>
              </li>
            ))}
          </ol>
        </header>

        <div className="wizard-body" aria-live="polite">
          {step === 0 && (
            <div className="wizard-step-content">
              <p className="wizard-step-hint">{WIZARD_STEPS[0].hint}</p>
              <div className="wizard-style-grid" role="radiogroup" aria-label="画风模板">
                {STYLE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    role="radio"
                    aria-checked={selectedStyle === option.value}
                    className={`wizard-style-card${selectedStyle === option.value ? ' selected' : ''}`}
                    onClick={() => setSelectedStyle(option.value)}
                  >
                    <span className={`wizard-style-swatch swatch-${option.value}`} aria-hidden="true" />
                    <strong>{option.label}</strong>
                    <span>{STYLE_DESCRIPTIONS[option.value] || ''}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {step === 1 && (
            <div className="wizard-step-content">
              <p className="wizard-step-hint">{WIZARD_STEPS[1].hint}</p>
              <div className="wizard-key-form">
                <div className="wizard-key-field">
                  <label htmlFor="wizard-mimo-key">MIMO_API_KEY · 剧本 / 分镜 LLM</label>
                  <div className="wizard-key-row">
                    <Input.Password
                      id="wizard-mimo-key"
                      value={keys.mimoApiKey}
                      onChange={(event) => updateKey('mimoApiKey', event.target.value)}
                      placeholder="sk-...（小米 MiMo 官方接口密钥）"
                      autoComplete="new-password"
                    />
                    <Button
                      icon={<ExperimentOutlined />}
                      loading={testing === 'llm'}
                      disabled={!modelCategories}
                      onClick={() => void handleTest('llm')}
                    >
                      测试
                    </Button>
                  </div>
                  {testFeedback.llm && (
                    <div className={`wizard-test-feedback ${testFeedback.llm.tone}`}>{testFeedback.llm.text}</div>
                  )}
                </div>
                <div className="wizard-key-field">
                  <label htmlFor="wizard-ark-key">ARK_API_KEY · 视频生成</label>
                  <div className="wizard-key-row">
                    <Input.Password
                      id="wizard-ark-key"
                      value={keys.arkApiKey}
                      onChange={(event) => updateKey('arkApiKey', event.target.value)}
                      placeholder="火山方舟 API Key（Seedream 图像 + SeedDance 视频共用）"
                      autoComplete="new-password"
                    />
                    <Button
                      icon={<ExperimentOutlined />}
                      loading={testing === 'video'}
                      disabled={!modelCategories}
                      onClick={() => void handleTest('video')}
                    >
                      测试
                    </Button>
                  </div>
                  {testFeedback.video && (
                    <div className={`wizard-test-feedback ${testFeedback.video.tone}`}>{testFeedback.video.text}</div>
                  )}
                </div>
              </div>
              <p className="wizard-key-note">暂不配置也可以：图像阶段会使用本地 PIL 占位图，先体验完整工作台流程。</p>
            </div>
          )}

          {step === 2 && (
            <div className="wizard-step-content">
              <p className="wizard-step-hint">{WIZARD_STEPS[2].hint}</p>
              <div className="wizard-entry-grid">
                <button
                  type="button"
                  className="wizard-entry-card"
                  onClick={() => void finish('sample')}
                  disabled={finishing != null}
                >
                  <FileImageOutlined className="wizard-entry-icon" />
                  <strong>打开示例项目</strong>
                  <span>零 Key 浏览内置短剧：分镜列表、角色场景素材与占位图故事板。</span>
                  <em className="wizard-entry-tag">{finishing === 'sample' ? '创建中…' : '推荐先看'}</em>
                </button>
                <button
                  type="button"
                  className="wizard-entry-card"
                  onClick={() => void finish('blank')}
                  disabled={finishing != null}
                >
                  <PlayCircleOutlined className="wizard-entry-icon" />
                  <strong>从空白开始</strong>
                  <span>用所选画风创建一个新项目，粘贴或上传剧本文本即可开始创作。</span>
                  <em className="wizard-entry-tag">{finishing === 'blank' ? '创建中…' : styleLabel(selectedStyle)}</em>
                </button>
              </div>
              <p className="wizard-key-note">{WIZARD_KEY_NOTE}</p>
            </div>
          )}
        </div>

        <footer className="wizard-footer">
          <Button onClick={() => void finish('skip')} disabled={finishing != null}>
            跳过向导
          </Button>
          <div className="wizard-footer-nav">
            <Button disabled={prevStep == null || finishing != null} onClick={handlePrev}>
              上一步
            </Button>
            {step < 2 && (
              <Button type="primary" disabled={!canNext || finishing != null} onClick={() => void handleNext()}>
                下一步
              </Button>
            )}
          </div>
        </footer>
      </div>
    </div>
  )
}

function styleLabel(value: string): string {
  return STYLE_OPTIONS.find((option) => option.value === value)?.label || value
}

export default FirstRunWizard
