import type React from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiOutlined,
  ArrowLeftOutlined,
  BgColorsOutlined,
  CopyOutlined,
  DollarOutlined,
  ExperimentOutlined,
  ReloadOutlined,
  SaveOutlined,
  UploadOutlined,
} from '@ant-design/icons'
import AutoComplete from 'antd/es/auto-complete'
import Button from 'antd/es/button'
import Input from 'antd/es/input'
import InputNumber from 'antd/es/input-number'
import message from 'antd/es/message'
import Segmented from 'antd/es/segmented'
import Select from 'antd/es/select'
import Slider from 'antd/es/slider'
import Switch from 'antd/es/switch'
import { projectApi, settingsApi } from '../services/api'
import PricingConfigPanel from './PricingConfigPanel'
import {
  DEFAULT_TEMPLATE,
  SYSTEM_PROMPT_MAX_LENGTH,
  cloneTemplate,
  importSkillTemplate,
  templateCopyPayload,
  type AgentSkillConfig,
  type SkillTemplate,
} from './skillTemplateModel'
import { STYLE_TEMPLATES_UPDATED_EVENT } from '../constants/events'
import { STYLE_DESCRIPTIONS, STYLE_OPTIONS } from '../constants/styleTemplates'
import { useProjectStore } from '../stores/projectStore'
import { applyThemeToDocument, getInitialTheme, notifyThemeChanged, THEME_OPTIONS, type AppTheme } from '../theme'

const { TextArea } = Input
const { Password } = Input

type StyleOption = { value: string; label: string; keywords?: string; custom?: boolean }

type ModelCategory = 'script' | 'image' | 'video' | 'voice'
type ModelConfig = Record<string, any>
type ModelConfigState = Record<ModelCategory, ModelConfig>
type DiscoveredModel = {
  id: string
  label: string
  owned_by?: string
  capabilities?: { audio_modes?: string[] }
}
type DiscoveredModelState = Record<ModelCategory, DiscoveredModel[]>

// 协议即代码路径：script 固定 openai-chat 不露出；其余类别提供下拉选择。
// 接入同协议新服务只需改 base_url/api_key/model；全新协议需后端新增适配器。
const PROTOCOL_OPTIONS: Partial<Record<ModelCategory, { value: string; label: string }[]>> = {
  image: [
    { value: 'ark-seedream', label: '火山方舟 Seedream（支持参考图）' },
    { value: 'qwen-image', label: '阿里云百炼 Qwen-Image（不支持参考图）' },
    { value: 'stability', label: 'Stability AI（不支持参考图）' },
    { value: 'placeholder', label: '本地占位图（离线/免密钥）' },
  ],
  video: [
    { value: 'ark-seedance', label: '火山方舟 Seedance（首帧参考 + 无声视频 + TTS，固定约 5 秒）' },
    { value: 'dashscope-wanx', label: '阿里云百炼 通义万相（首帧参考 + 无声视频 + TTS）' },
    { value: 'native-audio', label: '原生音视频（对白直出，厂商接入中）' },
  ],
  voice: [
    { value: 'mimo-tts', label: 'Mimo 内置 TTS' },
    { value: 'tencent-tts', label: '腾讯云语音合成（Key 填 SecretId:SecretKey）' },
    { value: 'dashscope-tts', label: '阿里云百炼 CosyVoice 语音合成' },
  ],
}

const AUDIO_MODE_OPTIONS = [
  { value: 'tts', label: 'TTS 配音合成（默认，无声视频 + 配音）' },
  { value: 'native', label: '原生音频（模型直接生成对白语音）' },
  { value: 'auto', label: '智能 auto（特写台词镜头用原生，其余 TTS）' },
]

const _SHOT_AUDIO_MODE_OPTIONS = [
  { value: '', label: '继承全局设置' },
  { value: 'tts', label: 'TTS 配音合成' },
  { value: 'native', label: '原生音频（需模型支持）' },
  { value: 'auto', label: '智能 auto' },
]

// 厂商预设只做表单快捷填充，不产生任何代码分支。
const VENDOR_PRESETS: Record<ModelCategory, { label: string; patch: ModelConfig }[]> = {
  script: [
    { label: 'OpenAI', patch: { base_url: 'https://api.openai.com/v1', model: 'gpt-4o', auth_style: 'bearer' } },
    {
      label: 'DeepSeek V4.1 Flash',
      patch: { base_url: 'https://api.deepseek.com', model: 'deepseek-flash', auth_style: 'bearer' },
    },
    {
      label: '小米 MiMo',
      patch: { base_url: 'https://token-plan-cn.xiaomimimo.com/v1', model: 'mimo-v2.5', auth_style: 'api-key-header' },
    },
    { label: '硅基流动', patch: { base_url: 'https://api.siliconflow.cn/v1', model: '', auth_style: 'bearer' } },
  ],
  image: [
    {
      label: '火山 Seedream 5.0 Pro',
      patch: {
        protocol: 'ark-seedream',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedream-5-0-pro-260628',
        image_size: '1440x2560',
      },
    },
    {
      label: '火山 Seedream 5.0 Flash',
      patch: {
        protocol: 'ark-seedream',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedream-5-0-flash-260915',
        image_size: '1440x2560',
      },
    },
    {
      label: '火山 Seedream 5.0 Lite',
      patch: {
        protocol: 'ark-seedream',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedream-5-0-lite-260128',
        image_size: '1440x2560',
      },
    },
    {
      label: '火山 Seedream 4.5',
      patch: {
        protocol: 'ark-seedream',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedream-4-5-251128',
        image_size: '1440x2560',
      },
    },
    {
      label: '阿里百炼 Qwen-Image 3.0 Pro',
      patch: {
        protocol: 'qwen-image',
        base_url: 'https://dashscope.aliyuncs.com/api/v1',
        model: 'qwen-image-3.0-pro',
        image_size: '1440x2560',
      },
    },
    {
      label: '阿里百炼 Qwen-Image 3.0',
      patch: {
        protocol: 'qwen-image',
        base_url: 'https://dashscope.aliyuncs.com/api/v1',
        model: 'qwen-image-3.0',
        image_size: '1440x2560',
      },
    },
    { label: 'Stability', patch: { protocol: 'stability', base_url: 'https://api.stability.ai/v2beta', model: '' } },
    { label: '本地占位图', patch: { protocol: 'placeholder' } },
  ],
  video: [
    {
      label: '火山 Seedance 2.5',
      patch: {
        protocol: 'ark-seedance',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedance-2-5-260628',
      },
    },
    {
      label: '火山 Seedance 2.0',
      patch: {
        protocol: 'ark-seedance',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedance-2-0-260128',
      },
    },
    {
      label: '火山 Seedance 2.0 Fast',
      patch: {
        protocol: 'ark-seedance',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedance-2-0-fast-260128',
      },
    },
    {
      label: '火山 Seedance 2.0 Mini',
      patch: {
        protocol: 'ark-seedance',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedance-2-0-mini-260615',
      },
    },
    {
      label: '火山 Seedance 1.0 Pro',
      patch: {
        protocol: 'ark-seedance',
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
        model: 'doubao-seedance-1-0-pro-250528',
      },
    },
    {
      label: '阿里百炼 Wan3.0',
      patch: {
        protocol: 'dashscope-wanx',
        base_url: 'https://dashscope.aliyuncs.com/api/v1',
        model: 'wan3.0-video',
      },
    },
    {
      label: '阿里百炼 Wan3.0 Prime',
      patch: {
        protocol: 'dashscope-wanx',
        base_url: 'https://dashscope.aliyuncs.com/api/v1',
        model: 'wan3.0-video-prime',
      },
    },
    {
      label: '阿里百炼 Wan2.7 图生视频',
      patch: {
        protocol: 'dashscope-wanx',
        base_url: 'https://dashscope.aliyuncs.com/api/v1',
        model: 'wan2.7-i2v',
      },
    },
    {
      label: '阿里百炼 Wan2.6 图生视频',
      patch: {
        protocol: 'dashscope-wanx',
        base_url: 'https://dashscope.aliyuncs.com/api/v1',
        model: 'wan2.6-i2v',
      },
    },
  ],
  voice: [
    {
      label: '小米 MiMo TTS',
      patch: {
        protocol: 'mimo-tts',
        base_url: 'https://token-plan-cn.xiaomimimo.com/v1',
        model: 'mimo-v2.5-tts',
        voice: '冰糖',
        format: 'wav',
      },
    },
    {
      label: '腾讯云 TTS',
      patch: {
        protocol: 'tencent-tts',
        base_url: 'https://tts.tencentcloudapi.com',
        model: '',
        voice: '101001',
        format: 'wav',
      },
    },
    {
      label: '阿里百炼 CosyVoice',
      patch: {
        protocol: 'dashscope-tts',
        base_url: 'https://dashscope.aliyuncs.com/api/v1',
        model: 'cosyvoice-v2',
        voice: 'longwan_v2',
        format: 'wav',
      },
    },
  ],
}

interface SystemSettingsPageProps {
  onBack: () => void
}

type SettingsTab = 'appearance' | 'models' | 'pricing' | 'skill'

const EMPTY_MODEL_CONFIG: ModelConfigState = {
  script: {},
  image: {},
  video: {},
  voice: {},
}

const EMPTY_DISCOVERED_MODELS: DiscoveredModelState = {
  script: [],
  image: [],
  video: [],
  voice: [],
}

const TAB_ITEMS: { key: SettingsTab; label: string; desc: string; icon: React.ReactNode }[] = [
  { key: 'appearance', label: '外观与画风', desc: '画风模板与全局生成规格', icon: <BgColorsOutlined /> },
  { key: 'models', label: '模型与 API 配置', desc: '剧本 / 图像 / 视频 / 配音接口', icon: <ApiOutlined /> },
  { key: 'pricing', label: '模型价格', desc: '单价、计价单位与分辨率倍率', icon: <DollarOutlined /> },
  { key: 'skill', label: 'Skill 配置', desc: '子 Agent 生成策略与绑定', icon: <ExperimentOutlined /> },
]

const SystemSettingsPage: React.FC<SystemSettingsPageProps> = ({ onBack }) => {
  const { projectId, parentProjectId, projectType, style, platform, outputFormat, resolution, setProject } =
    useProjectStore()
  const [activeTab, setActiveTab] = useState<SettingsTab>('appearance')
  const [appTheme, setAppTheme] = useState<AppTheme>(() => getInitialTheme())
  const [styleTemplates, setStyleTemplates] = useState<StyleOption[]>(STYLE_OPTIONS)
  const [customStyleName, setCustomStyleName] = useState('')
  const [customStyleKeywords, setCustomStyleKeywords] = useState('')
  const [customStyleNegative, setCustomStyleNegative] = useState('')
  const [savingStyleTemplate, setSavingStyleTemplate] = useState(false)
  const [skillTemplates, setSkillTemplates] = useState<SkillTemplate[]>([DEFAULT_TEMPLATE])
  const [activeSkillId, setActiveSkillId] = useState('default')
  const [draftSkill, setDraftSkill] = useState<SkillTemplate>(DEFAULT_TEMPLATE)
  const [globalDefaultSkillId, setGlobalDefaultSkillId] = useState('default')
  const [projectBindings, setProjectBindings] = useState<Record<string, string>>({})
  const [episodeBindings, setEpisodeBindings] = useState<Record<string, string>>({})
  const [savingSkill, setSavingSkill] = useState(false)
  const [modelConfig, setModelConfig] = useState<ModelConfigState>(EMPTY_MODEL_CONFIG)
  const [discoveredModels, setDiscoveredModels] = useState<DiscoveredModelState>(EMPTY_DISCOVERED_MODELS)
  const [discoveringCategory, setDiscoveringCategory] = useState<ModelCategory | null>(null)
  const [savingModel, setSavingModel] = useState(false)
  const skillImportRef = useRef<HTMLInputElement>(null)

  const skillOptions = useMemo(
    () => skillTemplates.map((item) => ({ value: item.id, label: item.name || item.id })),
    [skillTemplates],
  )

  const projectBindingKey = projectType === 'episode' ? parentProjectId : projectId || ''
  const projectBindingValue = projectBindingKey ? projectBindings[projectBindingKey] || '' : ''
  const episodeBindingValue = projectId ? episodeBindings[projectId] || '' : ''

  const loadStyleTemplates = async () => {
    const result = await settingsApi.styleTemplates()
    if (Array.isArray(result.templates)) {
      setStyleTemplates(result.templates)
    }
  }

  const loadSkillConfigs = async () => {
    const result = await settingsApi.skillConfigs()
    const templates = Array.isArray(result.templates) && result.templates.length ? result.templates : [DEFAULT_TEMPLATE]
    setSkillTemplates(templates)
    setGlobalDefaultSkillId(result.global_default_template_id || templates[0].id)
    setProjectBindings(result.project_bindings || {})
    setEpisodeBindings(result.episode_bindings || {})
    const initial =
      templates.find((item: SkillTemplate) => item.id === (result.global_default_template_id || 'default')) ||
      templates[0]
    setActiveSkillId(initial.id)
    setDraftSkill(cloneTemplate(initial))
  }

  const loadModelConfigs = async () => {
    const result = await settingsApi.modelConfigs()
    const categories = result.categories || {}
    setModelConfig({
      script: categories.script || {},
      image: categories.image || {},
      video: categories.video || {},
      voice: categories.voice || {},
    })
  }

  const updateProjectField = async (field: 'style' | 'resolution' | 'outputFormat', value: string) => {
    const previousValue = { style, resolution, outputFormat }[field]
    setProject({ [field]: value } as any)
    if (!projectId) return

    // store 用 camelCase，后端项目接口用 snake_case，这里做字段名映射。
    const apiFieldMap: Record<string, string> = {
      style: 'style',
      resolution: 'resolution',
      outputFormat: 'output_format',
    }
    try {
      await projectApi.update(projectId, { [apiFieldMap[field]]: value })
    } catch (err: any) {
      setProject({ [field]: previousValue } as any)
      message.error('项目配置更新失败：' + (err.message || '未知错误'))
    }
  }

  const handleThemeChange = (value: AppTheme) => {
    setAppTheme(value)
    applyThemeToDocument(value)
    notifyThemeChanged(value)
  }

  const handleSaveCustomStyle = async () => {
    const label = customStyleName.trim()
    const keywords = customStyleKeywords.trim()
    if (!label || !keywords) {
      message.warning('请输入模板名称和画风关键词')
      return
    }

    try {
      setSavingStyleTemplate(true)
      const template = await settingsApi.createStyleTemplate({
        label,
        keywords,
        negative_prompt: customStyleNegative.trim(),
      })
      setStyleTemplates((items) => [...items.filter((item) => item.value !== template.value), template])
      setDraftSkill((current) => ({
        ...current,
        script_agent: { ...current.script_agent, style_template_id: template.value },
        storyboard_agent: { ...current.storyboard_agent, style_template_id: template.value },
      }))
      await updateProjectField('style', template.value)
      window.dispatchEvent(new CustomEvent(STYLE_TEMPLATES_UPDATED_EVENT))
      setCustomStyleName('')
      setCustomStyleKeywords('')
      setCustomStyleNegative('')
      message.success('自定义画风模板已保存')
    } catch (err: any) {
      message.error('保存画风模板失败：' + (err.response?.data?.detail || err.message || '未知错误'))
    } finally {
      setSavingStyleTemplate(false)
    }
  }

  const handleSelectSkillTemplate = (templateId: string) => {
    const template = skillTemplates.find((item) => item.id === templateId)
    if (!template) return
    setActiveSkillId(template.id)
    setDraftSkill(cloneTemplate(template))
  }

  const handleSaveSkillTemplate = async (saveAs = false) => {
    const name = draftSkill.name.trim()
    if (!name) {
      message.warning('请输入 Skill 方案名称')
      return
    }
    const payload = saveAs ? templateCopyPayload(draftSkill) : { ...draftSkill, name }
    try {
      setSavingSkill(true)
      const saved = await settingsApi.saveSkillConfig(payload)
      setSkillTemplates((items) => [...items.filter((item) => item.id !== saved.id), saved])
      setActiveSkillId(saved.id)
      setDraftSkill(cloneTemplate(saved))
      message.success(saveAs ? 'Skill 方案已另存为模板' : 'Skill 方案已保存')
    } catch (err: any) {
      message.error('保存 Skill 方案失败：' + (err.response?.data?.detail || err.message || '未知错误'))
    } finally {
      setSavingSkill(false)
    }
  }

  const handleResetSkillTemplate = () => {
    const template = skillTemplates.find((item) => item.id === activeSkillId) || DEFAULT_TEMPLATE
    setDraftSkill(cloneTemplate(template))
    message.success('已恢复为上次保存的 Skill 方案')
  }

  const handleImportSkillFile = async (file: File) => {
    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      const imported = importSkillTemplate(parsed, draftSkill)
      if (!imported) {
        message.error('配置文件格式不正确：缺少 script_agent / storyboard_agent')
        return
      }
      setDraftSkill(imported)
      message.success('已导入 Skill 配置方案，请确认后点击保存生效')
    } catch (err: any) {
      message.error('导入失败：' + (err.message || '文件解析错误'))
    }
  }

  const handleSaveBindings = async (next?: {
    global_default_template_id?: string
    project_bindings?: Record<string, string>
    episode_bindings?: Record<string, string>
  }) => {
    const payload = {
      global_default_template_id: next?.global_default_template_id ?? globalDefaultSkillId,
      project_bindings: next?.project_bindings ?? projectBindings,
      episode_bindings: next?.episode_bindings ?? episodeBindings,
    }
    const result = await settingsApi.updateSkillBindings(payload)
    setGlobalDefaultSkillId(result.global_default_template_id)
    setProjectBindings(result.project_bindings || {})
    setEpisodeBindings(result.episode_bindings || {})
    message.success('Skill 绑定方案已保存')
  }

  const updateModelField = (category: ModelCategory, field: string, value: any) => {
    setModelConfig((current) => ({
      ...current,
      [category]: { ...current[category], [field]: value },
    }))
    if (field === 'base_url' || field === 'protocol') {
      setDiscoveredModels((current) => ({ ...current, [category]: [] }))
    }
  }

  const handleDiscoverModels = async (category: ModelCategory) => {
    const config = modelConfig[category]
    if (!String(config.base_url || '').trim()) {
      message.warning('请先填写 Base URL，再获取模型')
      return
    }
    try {
      setDiscoveringCategory(category)
      const result = await settingsApi.discoverModels({
        category,
        base_url: String(config.base_url || ''),
        // GET /model-configs masks persisted secrets. An empty value tells the
        // server to reuse the saved key for this unchanged endpoint.
        api_key: config.api_key === '********' ? '' : String(config.api_key || ''),
        protocol: String(config.protocol || config.provider || ''),
        auth_style: String(config.auth_style || 'bearer'),
      })
      const models = Array.isArray(result.models) ? (result.models as DiscoveredModel[]) : []
      setDiscoveredModels((current) => ({ ...current, [category]: models }))
      if (models.length && !String(config.model || '').trim()) {
        updateModelField(category, 'model', models[0].id)
      }
      message.success(`已获取 ${models.length} 个可用模型`)
    } catch (err: any) {
      message.error('获取模型失败：' + (err.response?.data?.detail || err.message || '请检查地址和密钥'))
    } finally {
      setDiscoveringCategory(null)
    }
  }

  const handleSaveModelConfig = async () => {
    try {
      setSavingModel(true)
      const result = await settingsApi.saveModelConfigs({ categories: modelConfig })
      const categories = result.categories || {}
      setModelConfig({
        script: categories.script || {},
        image: categories.image || {},
        video: categories.video || {},
        voice: categories.voice || {},
      })
      message.success('模型与 API 配置已保存，新任务将自动生效')
    } catch (err: any) {
      message.error('保存模型配置失败：' + (err.response?.data?.detail || err.message || '未知错误'))
    } finally {
      setSavingModel(false)
    }
  }

  useEffect(() => {
    void loadStyleTemplates().catch(() => undefined)
    void loadSkillConfigs().catch(() => undefined)
    void loadModelConfigs().catch(() => undefined)
  }, [])

  return (
    <section className="main-workspace settings-workspace" aria-label="系统设置">
      <div className="settings-page panel-enter">
        <div className="settings-page-head">
          <div>
            <div className="asset-board-title">系统设置</div>
            <div className="asset-board-note">
              画风模板、模型 API 与子 Agent Skill 方案集中管理，保存后只影响后续新生成素材。
            </div>
          </div>
          <Button icon={<ArrowLeftOutlined />} onClick={onBack}>
            返回工作区
          </Button>
        </div>

        <div className="settings-shell">
          <nav className="settings-nav" aria-label="设置分区">
            {TAB_ITEMS.map((tab) => (
              <button
                key={tab.key}
                type="button"
                className={`settings-nav-item${activeTab === tab.key ? ' active' : ''}`}
                onClick={() => setActiveTab(tab.key)}
              >
                <span className="settings-nav-icon">{tab.icon}</span>
                <span className="settings-nav-text">
                  <strong>{tab.label}</strong>
                  <span>{tab.desc}</span>
                </span>
              </button>
            ))}
          </nav>

          <div className="settings-content">
            {activeTab === 'appearance' && (
              <div className="settings-page-grid">
                <section className="settings-section settings-section-wide theme-settings-section">
                  <div className="settings-section-title">界面主题</div>
                  <div className="theme-switch-row">
                    <Segmented
                      value={appTheme}
                      onChange={(value) => handleThemeChange(value as AppTheme)}
                      options={THEME_OPTIONS.map((item) => ({ value: item.value, label: item.label }))}
                    />
                    <span>{THEME_OPTIONS.find((item) => item.value === appTheme)?.description}</span>
                  </div>
                </section>

                <section className="settings-section settings-section-wide style-preset-section">
                  <div className="settings-section-title">预设与自定义模板</div>
                  <div className="settings-field">
                    <span>当前画风模板</span>
                    <Select
                      value={style}
                      onChange={(value) => void updateProjectField('style', value)}
                      options={styleTemplates.map((item) => ({
                        value: item.value,
                        label: item.custom ? `${item.label}（自定义）` : item.label,
                      }))}
                    />
                  </div>
                  <div className="settings-template-list">
                    {styleTemplates.map((item) => (
                      <button
                        key={item.value}
                        type="button"
                        className={`settings-template-chip${style === item.value ? ' active' : ''}`}
                        onClick={() => void updateProjectField('style', item.value)}
                      >
                        <strong>{item.label}</strong>
                        <span>
                          {item.custom
                            ? item.keywords || '自定义模板'
                            : STYLE_DESCRIPTIONS[item.value] || item.keywords || '预设模板'}
                        </span>
                      </button>
                    ))}
                  </div>
                </section>

                <section className="settings-section custom-style-section">
                  <div className="settings-section-title">保存自定义画风</div>
                  <div className="settings-field">
                    <span>模板名称</span>
                    <Input value={customStyleName} onChange={(event) => setCustomStyleName(event.target.value)} />
                  </div>
                  <div className="settings-field">
                    <span>画风关键词</span>
                    <TextArea
                      autoSize={{ minRows: 5, maxRows: 9 }}
                      value={customStyleKeywords}
                      onChange={(event) => setCustomStyleKeywords(event.target.value)}
                      placeholder="例如：柔和水彩、低饱和校园光、干净线稿、统一暖色后期"
                    />
                  </div>
                  <div className="settings-field">
                    <span>负向关键词</span>
                    <Input
                      value={customStyleNegative}
                      onChange={(event) => setCustomStyleNegative(event.target.value)}
                      placeholder="例如：过曝、凌乱笔触、文字、水印"
                    />
                  </div>
                  <Button
                    type="primary"
                    icon={<SaveOutlined />}
                    loading={savingStyleTemplate}
                    onClick={() => void handleSaveCustomStyle()}
                  >
                    保存为自定义模板
                  </Button>
                </section>

                <section className="settings-section generation-settings-section">
                  <div className="settings-section-title">全局生成配置</div>
                  <div className="settings-field">
                    <span>输出分辨率</span>
                    <Select
                      value={resolution}
                      onChange={(value) => void updateProjectField('resolution', value)}
                      options={[
                        { value: '720p', label: '高清 720' },
                        { value: '1080p', label: '全高清 1080' },
                        { value: '2k', label: '影院级 2K' },
                        { value: '4k', label: '超高清 4K' },
                      ]}
                    />
                  </div>
                  <div className="settings-field">
                    <span>发布平台</span>
                    <Input value={platform} disabled />
                  </div>
                  <div className="settings-field">
                    <span>画面比例</span>
                    <Select
                      value={outputFormat}
                      onChange={(value) => void updateProjectField('outputFormat', value)}
                      options={[
                        { value: '9:16', label: '竖屏 9:16（抖音 / 快手）' },
                        { value: '16:9', label: '横屏 16:9（横版长视频）' },
                        { value: '1:1', label: '方形 1:1（信息流）' },
                        { value: '4:3', label: '传统 4:3' },
                        { value: '3:4', label: '竖版 3:4' },
                      ]}
                    />
                  </div>
                </section>
              </div>
            )}

            {activeTab === 'models' && (
              <div className="settings-models">
                <div className="settings-models-head">
                  <div>
                    <div className="settings-section-title">模型与 API 自定义配置</div>
                    <div className="asset-board-note">
                      分别配置剧本、图像、视频、配音四类生成模型的接口地址、密钥与模型名，保存后新任务自动加载最新配置。
                    </div>
                  </div>
                  <Button
                    type="primary"
                    icon={<SaveOutlined />}
                    loading={savingModel}
                    onClick={() => void handleSaveModelConfig()}
                  >
                    保存全部配置
                  </Button>
                </div>
                <div className="settings-model-grid">
                  <ModelConfigCard
                    title="剧本生成模型"
                    subtitle="LLM · 脚本生成 / 分镜决策 · OpenAI 兼容协议"
                    category="script"
                    config={modelConfig.script}
                    onChange={updateModelField}
                    discoveredModels={discoveredModels.script}
                    discovering={discoveringCategory === 'script'}
                    onDiscover={handleDiscoverModels}
                    extraFields={[{ key: 'max_tokens', label: '最大 Token', type: 'number' }]}
                    showAuthStyle
                  />
                  <ModelConfigCard
                    title="图像生成模型"
                    subtitle="角色三视图 / 场景基准 / 定稿故事板"
                    category="image"
                    config={modelConfig.image}
                    onChange={updateModelField}
                    discoveredModels={discoveredModels.image}
                    discovering={discoveringCategory === 'image'}
                    onDiscover={handleDiscoverModels}
                    extraFields={[
                      { key: 'image_size', label: '出图尺寸', type: 'text', placeholder: '例如 1440x2560' },
                    ]}
                  />
                  <ModelConfigCard
                    title="视频生成模型"
                    subtitle="逐镜头视频 · 音频路径按能力自动路由"
                    category="video"
                    config={modelConfig.video}
                    onChange={updateModelField}
                    discoveredModels={discoveredModels.video}
                    discovering={discoveringCategory === 'video'}
                    onDiscover={handleDiscoverModels}
                    showAudioMode
                  />
                  <ModelConfigCard
                    title="配音生成模型"
                    subtitle="TTS 语音合成"
                    category="voice"
                    config={modelConfig.voice}
                    onChange={updateModelField}
                    discoveredModels={discoveredModels.voice}
                    discovering={discoveringCategory === 'voice'}
                    onDiscover={handleDiscoverModels}
                    extraFields={[
                      { key: 'voice', label: '默认音色', type: 'text', placeholder: '例如 冰糖、101001 或 longwan_v2' },
                      { key: 'format', label: '音频格式', type: 'text', placeholder: '例如 wav' },
                    ]}
                  />
                </div>
              </div>
            )}

            {activeTab === 'pricing' && <PricingConfigPanel />}

            {activeTab === 'skill' && (
              <section className="settings-section skill-settings-section">
                <div className="settings-section-title">子 Agent Skill 配置</div>
                <div className="skill-template-toolbar">
                  <Select value={activeSkillId} onChange={handleSelectSkillTemplate} options={skillOptions} />
                  <Input
                    value={draftSkill.name}
                    onChange={(event) => setDraftSkill({ ...draftSkill, name: event.target.value })}
                  />
                  <Button
                    type="primary"
                    icon={<SaveOutlined />}
                    loading={savingSkill}
                    onClick={() => void handleSaveSkillTemplate(false)}
                  >
                    保存
                  </Button>
                  <Button icon={<ReloadOutlined />} onClick={handleResetSkillTemplate}>
                    重置
                  </Button>
                  <Button
                    icon={<CopyOutlined />}
                    loading={savingSkill}
                    onClick={() => void handleSaveSkillTemplate(true)}
                  >
                    另存为模板
                  </Button>
                  <Button icon={<UploadOutlined />} onClick={() => skillImportRef.current?.click()}>
                    导入配置
                  </Button>
                  <input
                    ref={skillImportRef}
                    type="file"
                    accept="application/json,.json"
                    style={{ display: 'none' }}
                    onChange={(event) => {
                      const file = event.target.files?.[0]
                      if (file) void handleImportSkillFile(file)
                      event.target.value = ''
                    }}
                  />
                </div>
                <div className="skill-binding-grid">
                  <BindingSelect
                    label="全局默认方案"
                    value={globalDefaultSkillId}
                    options={skillOptions}
                    onChange={(value) => {
                      setGlobalDefaultSkillId(value)
                      void handleSaveBindings({ global_default_template_id: value })
                    }}
                  />
                  <BindingSelect
                    label="大项目绑定方案"
                    value={projectBindingValue}
                    options={[{ value: '', label: '继承全局默认' }, ...skillOptions]}
                    disabled={!projectBindingKey}
                    onChange={(value) => {
                      const next = { ...projectBindings }
                      if (projectBindingKey) {
                        if (value) next[projectBindingKey] = value
                        else delete next[projectBindingKey]
                      }
                      setProjectBindings(next)
                      void handleSaveBindings({ project_bindings: next })
                    }}
                  />
                  <BindingSelect
                    label="单剧集绑定方案"
                    value={episodeBindingValue}
                    options={[{ value: '', label: '继承项目或全局' }, ...skillOptions]}
                    disabled={!projectId}
                    onChange={(value) => {
                      const next = { ...episodeBindings }
                      if (projectId) {
                        if (value) next[projectId] = value
                        else delete next[projectId]
                      }
                      setEpisodeBindings(next)
                      void handleSaveBindings({ episode_bindings: next })
                    }}
                  />
                </div>
                <div className="skill-agent-grid">
                  <AgentSkillPanel
                    title="剧本生成子Agent配置区"
                    agentKey="script_agent"
                    agent={draftSkill.script_agent}
                    styleTemplates={styleTemplates}
                    onChange={(agent) => setDraftSkill((current) => ({ ...current, script_agent: agent }))}
                  />
                  <AgentSkillPanel
                    title="分镜生成子Agent配置区"
                    agentKey="storyboard_agent"
                    agent={draftSkill.storyboard_agent}
                    styleTemplates={styleTemplates}
                    onChange={(agent) => setDraftSkill((current) => ({ ...current, storyboard_agent: agent }))}
                  />
                </div>
              </section>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

// Provider 能力徽标：数据来自后端适配器 capabilities 声明（实时、如实），
// 不在前端硬编码任何能力，避免界面宣称不存在的能力。
function CapabilityBadges({
  category,
  protocol,
  capabilities,
}: {
  category: ModelCategory
  protocol: string
  capabilities?: Record<string, any>
}) {
  if (!capabilities || !protocol) return null
  const badges: string[] = []
  if (category === 'image') {
    badges.push(
      capabilities.reference_images
        ? '支持参考图（多图）'
        : '不支持参考图（需参考图的阶段会降级为纯文本生成并明确告警，绝不假装参考图已生效）',
    )
    if (capabilities.requires_credentials === false) badges.push('免密钥（本地占位图）')
  }
  if (category === 'video') {
    badges.push(
      capabilities.reference_image
        ? `首帧参考：${capabilities.reference_mode || 'first_frame_only'}`
        : '不支持首帧参考图',
    )
    if (capabilities.fixed_duration) badges.push(`固定时长 ${capabilities.fixed_duration}s（超长镜头需拆分）`)
    badges.push(capabilities.native_audio ? '原生音频' : '无声视频（对白走 TTS 配音链路）')
  }
  if (!badges.length) return null
  return (
    <div
      className="model-config-capabilities"
      style={{ display: 'flex', flexWrap: 'wrap', gap: 6, margin: '4px 0 10px' }}
    >
      {badges.map((badge) => (
        <span
          key={badge}
          style={{
            fontSize: 12,
            padding: '2px 8px',
            borderRadius: 10,
            background: 'rgba(127,127,127,0.14)',
            color: 'var(--ant-color-text-secondary, #888)',
          }}
        >
          {badge}
        </span>
      ))}
    </div>
  )
}

function ModelConfigCard({
  title,
  subtitle,
  category,
  config,
  onChange,
  extraFields,
  showAuthStyle,
  showAudioMode,
  discoveredModels,
  discovering,
  onDiscover,
}: {
  title: string
  subtitle: string
  category: ModelCategory
  config: ModelConfig
  onChange: (category: ModelCategory, field: string, value: any) => void
  extraFields?: { key: string; label: string; type: 'text' | 'number'; placeholder?: string }[]
  showAuthStyle?: boolean
  showAudioMode?: boolean
  discoveredModels: DiscoveredModel[]
  discovering: boolean
  onDiscover: (category: ModelCategory) => void
}) {
  const protocolOptions = PROTOCOL_OPTIONS[category]
  const currentProtocol = String(config.protocol || config.provider || '').toLowerCase()
  const audioMode = String(config.audio_mode || '').toLowerCase()
  const selectedModel = discoveredModels.find((item) => item.id === String(config.model || ''))
  const audioModes = selectedModel?.capabilities?.audio_modes || []
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const audioModeOptions = AUDIO_MODE_OPTIONS.map((option) => ({
    ...option,
    disabled: Boolean(audioModes.length) && option.value === 'native' && !audioModes.includes('native'),
  }))

  useEffect(() => {
    if (discoveredModels.length > 0) setModelDropdownOpen(true)
  }, [discoveredModels.length])
  return (
    <div className="model-config-card">
      <div className="model-config-head">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <CapabilityBadges category={category} protocol={currentProtocol} capabilities={config.capabilities} />
      <div className="settings-field">
        <span>厂商预设（点击填充表单）</span>
        <div className="settings-template-list">
          {VENDOR_PRESETS[category].map((preset) => (
            <button
              key={preset.label}
              type="button"
              className="settings-template-chip"
              onClick={() =>
                Object.entries(preset.patch).forEach(([field, value]) => {
                  onChange(category, field, value)
                })
              }
            >
              <strong>{preset.label}</strong>
              <span>{String(preset.patch.base_url || preset.patch.protocol || '快捷填充')}</span>
            </button>
          ))}
        </div>
      </div>
      {protocolOptions && (
        <div className="settings-field">
          <span>接入协议 Protocol</span>
          <Select
            value={currentProtocol || protocolOptions[0].value}
            onChange={(value) => onChange(category, 'protocol', value)}
            options={protocolOptions}
          />
        </div>
      )}
      {showAuthStyle && (
        <div className="settings-field">
          <span>鉴权方式</span>
          <Select
            value={String(config.auth_style || 'bearer').toLowerCase()}
            onChange={(value) => onChange(category, 'auth_style', value)}
            options={[
              { value: 'bearer', label: 'Bearer Token（Authorization 头）' },
              { value: 'api-key-header', label: 'api-key 请求头（小米 MiMo 等）' },
            ]}
          />
        </div>
      )}
      <div className="settings-field">
        <span>接口地址 Base URL</span>
        <Input
          value={config.base_url || ''}
          placeholder="https://..."
          onChange={(event) => onChange(category, 'base_url', event.target.value)}
        />
      </div>
      <div className="settings-field">
        <span>API 密钥</span>
        <Password
          value={config.api_key || ''}
          placeholder="留空表示沿用环境变量配置；换地址后需填新密钥"
          visibilityToggle
          onChange={(event) => onChange(category, 'api_key', event.target.value)}
        />
      </div>
      <div className="model-discovery-row">
        <Button icon={<ReloadOutlined />} loading={discovering} onClick={() => onDiscover(category)}>
          获取可用模型
        </Button>
        <span>
          {discoveredModels.length
            ? `已发现 ${discoveredModels.length} 个模型`
            : '输入地址和密钥后，从服务端读取模型列表'}
        </span>
      </div>
      <div className="settings-field">
        <span>模型名称</span>
        <AutoComplete
          value={config.model || ''}
          open={modelDropdownOpen && discoveredModels.length > 0}
          options={discoveredModels.map((item) => ({
            value: item.id,
            label: item.label === item.id ? item.id : `${item.label} · ${item.id}`,
          }))}
          onChange={(value) => onChange(category, 'model', value)}
          onFocus={() => {
            if (discoveredModels.length > 0) setModelDropdownOpen(true)
          }}
          onOpenChange={setModelDropdownOpen}
          onSelect={() => setModelDropdownOpen(false)}
          placeholder={discoveredModels.length ? '搜索或输入模型名称' : '可手动输入模型名称'}
          filterOption={(inputValue, option) =>
            String(option?.value || '')
              .toLowerCase()
              .includes(inputValue.toLowerCase())
          }
        />
      </div>
      {showAudioMode && selectedModel ? (
        <div className="model-capability-row" aria-label="模型能力">
          <span>模型能力</span>
          <div className="model-capability-tags">
            {audioModes.includes('native') && <span className="model-capability-tag">有声视频</span>}
            {audioModes.includes('silent') && <span className="model-capability-tag">无声视频</span>}
            {!audioModes.length && <span className="model-capability-tag unknown">未声明</span>}
          </div>
        </div>
      ) : null}
      {showAudioMode && (
        <>
          <div className="settings-field">
            <span>音频路径 audio_mode</span>
            <Select
              value={['tts', 'native', 'auto'].includes(audioMode) ? audioMode : 'tts'}
              onChange={(value) => onChange(category, 'audio_mode', value)}
              options={audioModeOptions}
            />
          </div>
          <div className="asset-board-note">
            镜头可在右侧镜头编辑面板单独覆盖音频方式（保存于镜头的 continuity_profile，优先于此处全局设置）。
          </div>
        </>
      )}
      {extraFields?.map((field) => (
        <div className="settings-field" key={field.key}>
          <span>{field.label}</span>
          {field.type === 'number' ? (
            <InputNumber
              style={{ width: '100%' }}
              value={config[field.key] ?? undefined}
              onChange={(value) => onChange(category, field.key, value)}
            />
          ) : (
            <Input
              value={config[field.key] || ''}
              placeholder={field.placeholder}
              onChange={(event) => onChange(category, field.key, event.target.value)}
            />
          )}
        </div>
      ))}
    </div>
  )
}

function BindingSelect({
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  label: string
  value: string
  options: { value: string; label: string }[]
  disabled?: boolean
  onChange: (value: string) => void
}) {
  return (
    <div className="settings-field skill-binding-field">
      <span>{label}</span>
      <Select value={value} disabled={disabled} onChange={onChange} options={options} />
    </div>
  )
}

function AgentSkillPanel({
  title,
  agentKey,
  agent,
  styleTemplates,
  onChange,
}: {
  title: string
  agentKey: 'script_agent' | 'storyboard_agent'
  agent: AgentSkillConfig
  styleTemplates: StyleOption[]
  onChange: (agent: AgentSkillConfig) => void
}) {
  const update = <K extends keyof AgentSkillConfig>(key: K, value: AgentSkillConfig[K]) => {
    onChange({ ...agent, [key]: value })
  }

  return (
    <div className="skill-agent-panel">
      <div className="skill-agent-title">{title}</div>
      <div className="settings-field">
        <span>Skill 画风模板</span>
        <Select
          value={agent.style_template_id}
          disabled={!agent.style_override_enabled}
          onChange={(value) => update('style_template_id', value)}
          options={[
            { value: '', label: '继承项目画风' },
            ...styleTemplates.map((item) => ({
              value: item.value,
              label: item.custom ? `${item.label}（自定义）` : item.label,
            })),
          ]}
        />
      </div>
      <ToggleRow
        label="允许 Skill 覆盖项目画风"
        checked={agent.style_override_enabled}
        onChange={(value) => update('style_override_enabled', value)}
      />
      <div className="settings-field">
        <span>自定义画风补充</span>
        <TextArea
          autoSize={{ minRows: 2, maxRows: 4 }}
          value={agent.custom_style_keywords}
          onChange={(event) => update('custom_style_keywords', event.target.value)}
        />
      </div>
      <SystemPromptField
        agentKey={agentKey}
        value={agent.system_prompt}
        onChange={(value) => update('system_prompt', value)}
      />
      <div className="settings-field">
        <span>镜头构图规范默认参数</span>
        <TextArea
          autoSize={{ minRows: 2, maxRows: 4 }}
          value={agent.camera_composition}
          onChange={(event) => update('camera_composition', event.target.value)}
        />
      </div>
      <ToggleRow
        label="TTS 过滤指令话术"
        checked={agent.filter_tts_instruction_text}
        onChange={(value) => update('filter_tts_instruction_text', value)}
      />
      <ToggleRow
        label="强制引用人物/场景基准图"
        checked={agent.force_character_scene_references}
        onChange={(value) => update('force_character_scene_references', value)}
      />
      <ToggleRow
        label="Prompt 自动拼装"
        checked={agent.prompt_auto_assembly}
        onChange={(value) => update('prompt_auto_assembly', value)}
      />
      <div className="skill-toggle-row">
        <span>OpenPose / Depth 控制</span>
        <span>未接入（unsupported）</span>
      </div>
      <ToggleRow
        label="镜头续帧连贯逻辑"
        checked={agent.continuity_enabled}
        onChange={(value) => update('continuity_enabled', value)}
      />
      <WeightField
        label="参考图画风权重"
        value={agent.style_reference_weight}
        onChange={(value) => update('style_reference_weight', value)}
      />
      <WeightField
        label="动作权重"
        value={agent.action_reference_weight}
        onChange={(value) => update('action_reference_weight', value)}
      />
    </div>
  )
}

// 系统提示词编辑框：多行输入，留空 = 使用后端内置默认提示词；JSON 输出
// 契约由后端固定拼装，用户提示词无法删除结构约束。计数按去首尾空白后的长度。
function SystemPromptField({
  agentKey,
  value,
  onChange,
}: {
  agentKey: 'script_agent' | 'storyboard_agent'
  value: string
  onChange: (value: string) => void
}) {
  const label = agentKey === 'script_agent' ? '剧本系统提示词' : '分镜系统提示词'
  const trimmed = value.trim()
  const overLimit = trimmed.length > SYSTEM_PROMPT_MAX_LENGTH
  return (
    <div className="settings-field skill-system-prompt-field">
      <span>
        {label}
        <em className="skill-system-prompt-count">
          （{trimmed.length}/{SYSTEM_PROMPT_MAX_LENGTH}
          {value ? '，留空恢复默认' : '，当前使用默认提示词'}）
        </em>
      </span>
      <TextArea
        autoSize={{ minRows: 4, maxRows: 12 }}
        value={value}
        maxLength={SYSTEM_PROMPT_MAX_LENGTH * 2}
        showCount={false}
        placeholder={
          agentKey === 'script_agent'
            ? '留空使用默认：你是漫剧编剧。请输出完整中文漫剧剧本，包含标题、人物、场景、动作、对白和情绪……\n可自定义角色定位、创作风格、语言语气；JSON 输出契约由系统固定附加，不可删除。'
            : '留空使用默认：你是专业漫剧分镜师。根据剧本场景输出可执行分镜 JSON……\n可自定义角色定位、创作风格、语言语气；JSON 输出契约由系统固定附加，不可删除。'
        }
        onChange={(event) => onChange(event.target.value)}
        status={overLimit ? 'error' : undefined}
      />
      <div className="skill-system-prompt-actions">
        <Button
          size="small"
          icon={<ReloadOutlined />}
          disabled={!value}
          onClick={() => onChange('')}
          title="清空自定义提示词，恢复后端默认系统提示词"
        >
          恢复默认提示词
        </Button>
        {overLimit ? (
          <span className="skill-system-prompt-error">
            提示词去空白后超过 {SYSTEM_PROMPT_MAX_LENGTH} 字符，保存会被拒绝
          </span>
        ) : null}
      </div>
    </div>
  )
}

function ToggleRow({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <div className="skill-toggle-row">
      <span>{label}</span>
      <Switch checked={checked} onChange={onChange} />
    </div>
  )
}

function WeightField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <div className="skill-weight-row">
      <span>{label}</span>
      <Slider min={0} max={1} step={0.05} value={value} onChange={onChange} />
      <InputNumber min={0} max={1} step={0.05} value={value} onChange={(next) => onChange(Number(next || 0))} />
    </div>
  )
}

export default SystemSettingsPage
