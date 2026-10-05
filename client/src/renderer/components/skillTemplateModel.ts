// Skill 方案（子 Agent 配置模板）的纯数据模型：类型、默认值、克隆与导入
// 归一化。SystemSettingsPage 只做展示与事件接线，字段语义集中在这里，
// 方便对 system_prompt 的保存/重置/导入行为做单测。

export type AgentSkillConfig = {
  style_template_id: string
  style_override_enabled: boolean
  custom_style_keywords: string
  system_prompt: string
  filter_tts_instruction_text: boolean
  camera_composition: string
  force_character_scene_references: boolean
  prompt_auto_assembly: boolean
  openpose_lock_enabled: boolean
  style_reference_weight: number
  action_reference_weight: number
  continuity_enabled: boolean
}

export type SkillTemplate = {
  id: string
  name: string
  script_agent: AgentSkillConfig
  storyboard_agent: AgentSkillConfig
}

export const SYSTEM_PROMPT_MAX_LENGTH = 20000

export const DEFAULT_AGENT_CONFIG: AgentSkillConfig = {
  style_template_id: '',
  style_override_enabled: false,
  custom_style_keywords: '',
  // 空字符串 = 使用后端内置默认系统提示词；清空输入框即恢复默认。
  system_prompt: '',
  filter_tts_instruction_text: true,
  camera_composition: 'medium shot, vertical 9:16, clear subject staging',
  force_character_scene_references: true,
  prompt_auto_assembly: true,
  openpose_lock_enabled: false,
  style_reference_weight: 0.45,
  action_reference_weight: 0.3,
  continuity_enabled: true,
}

export const DEFAULT_TEMPLATE: SkillTemplate = {
  id: 'default',
  name: '默认 Skill 方案',
  script_agent: DEFAULT_AGENT_CONFIG,
  storyboard_agent: DEFAULT_AGENT_CONFIG,
}

// 非法类型（导入文件里 system_prompt 是数字/对象等）统一回落空串，
// 与后端 _normalize_agent_config 的口径一致。
function normalizeAgentConfig(source: Record<string, any> | undefined | null): AgentSkillConfig {
  const merged: Record<string, any> = { ...DEFAULT_AGENT_CONFIG, ...(source || {}) }
  merged.system_prompt =
    typeof merged.system_prompt === 'string' ? merged.system_prompt.slice(0, SYSTEM_PROMPT_MAX_LENGTH) : ''
  merged.camera_composition = typeof merged.camera_composition === 'string' ? merged.camera_composition : ''
  merged.custom_style_keywords = typeof merged.custom_style_keywords === 'string' ? merged.custom_style_keywords : ''
  merged.style_template_id = typeof merged.style_template_id === 'string' ? merged.style_template_id : ''
  return merged as AgentSkillConfig
}

export function cloneTemplate(template: SkillTemplate): SkillTemplate {
  return {
    ...template,
    script_agent: normalizeAgentConfig(template.script_agent),
    storyboard_agent: normalizeAgentConfig(template.storyboard_agent),
  }
}

// 导入配置：接受完整方案或 { template: ... } 包裹结构；单 Agent 文件按
// 兼容策略填充另一个 Agent。system_prompt 随其它字段一起被读取和保留。
export function importSkillTemplate(parsed: any, current: SkillTemplate): SkillTemplate | null {
  const candidate = parsed?.script_agent || parsed?.storyboard_agent ? parsed : parsed?.template || parsed
  if (!candidate || (!candidate.script_agent && !candidate.storyboard_agent)) {
    return null
  }
  return cloneTemplate({
    id: current.id,
    name: candidate.name ? `${candidate.name}` : `${current.name}（导入）`,
    script_agent: { ...DEFAULT_AGENT_CONFIG, ...(candidate.script_agent || candidate.storyboard_agent || {}) },
    storyboard_agent: { ...DEFAULT_AGENT_CONFIG, ...(candidate.storyboard_agent || candidate.script_agent || {}) },
  })
}

// 另存为副本：只换 id/name，字段（含 system_prompt）原样保留。
export function templateCopyPayload(template: SkillTemplate): SkillTemplate {
  const name = template.name.trim()
  return {
    ...template,
    id: `${template.id}_${Date.now()}`,
    name: `${name} 副本`,
  }
}
