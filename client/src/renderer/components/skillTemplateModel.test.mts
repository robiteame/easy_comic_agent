import assert from 'node:assert/strict'
import test from 'node:test'

import {
  DEFAULT_AGENT_CONFIG,
  DEFAULT_TEMPLATE,
  SYSTEM_PROMPT_MAX_LENGTH,
  cloneTemplate,
  importSkillTemplate,
  templateCopyPayload,
} from './skillTemplateModel.ts'

// 覆盖需求：导入配置、另存为模板、重置配置都能正确保留或恢复
// system_prompt；空值/非法值回落默认，不把非法内容带进保存负载。

const CUSTOM_SCRIPT_PROMPT = '你是古风漫剧编剧，对白讲究韵律。\n多行第二段。'
const CUSTOM_STORYBOARD_PROMPT = '你是短视频分镜师，偏好快节奏切镜。'

function templateWithPrompts(scriptPrompt: string, storyboardPrompt: string) {
  return cloneTemplate({
    ...DEFAULT_TEMPLATE,
    script_agent: { ...DEFAULT_AGENT_CONFIG, system_prompt: scriptPrompt },
    storyboard_agent: { ...DEFAULT_AGENT_CONFIG, system_prompt: storyboardPrompt },
  })
}

test('cloneTemplate 保留 system_prompt 并为缺失字段补默认空串', () => {
  const legacy = {
    ...DEFAULT_TEMPLATE,
    script_agent: { ...DEFAULT_AGENT_CONFIG } as any,
    storyboard_agent: { ...DEFAULT_AGENT_CONFIG } as any,
  }
  delete (legacy.script_agent as any).system_prompt
  delete (legacy.storyboard_agent as any).system_prompt
  const cloned = cloneTemplate(legacy as any)
  assert.equal(cloned.script_agent.system_prompt, '')
  assert.equal(cloned.storyboard_agent.system_prompt, '')

  const custom = cloneTemplate(templateWithPrompts(CUSTOM_SCRIPT_PROMPT, CUSTOM_STORYBOARD_PROMPT) as any)
  assert.equal(custom.script_agent.system_prompt, CUSTOM_SCRIPT_PROMPT)
  assert.equal(custom.storyboard_agent.system_prompt, CUSTOM_STORYBOARD_PROMPT)
})

test('cloneTemplate 把非法类型的 system_prompt 归一化为空串', () => {
  const poisoned = cloneTemplate({
    ...DEFAULT_TEMPLATE,
    script_agent: { ...DEFAULT_AGENT_CONFIG, system_prompt: 123 } as any,
    storyboard_agent: { ...DEFAULT_AGENT_CONFIG, system_prompt: { hacked: true } } as any,
  } as any)
  assert.equal(poisoned.script_agent.system_prompt, '')
  assert.equal(poisoned.storyboard_agent.system_prompt, '')
})

test('导入配置读取 system_prompt：完整方案与单 Agent 文件都保留该字段', () => {
  const current = templateWithPrompts('', '')
  const imported = importSkillTemplate(
    {
      name: '导入方案',
      script_agent: { system_prompt: CUSTOM_SCRIPT_PROMPT },
      storyboard_agent: { system_prompt: CUSTOM_STORYBOARD_PROMPT },
    },
    current,
  )
  assert.ok(imported)
  assert.equal(imported!.name, '导入方案')
  assert.equal(imported!.script_agent.system_prompt, CUSTOM_SCRIPT_PROMPT)
  assert.equal(imported!.storyboard_agent.system_prompt, CUSTOM_STORYBOARD_PROMPT)

  // 旧版导出只包含 script_agent：兼容填充 storyboard，不丢 system_prompt。
  const singleAgent = importSkillTemplate({ script_agent: { system_prompt: CUSTOM_SCRIPT_PROMPT } }, current)
  assert.ok(singleAgent)
  assert.equal(singleAgent!.script_agent.system_prompt, CUSTOM_SCRIPT_PROMPT)
  assert.equal(singleAgent!.storyboard_agent.system_prompt, CUSTOM_SCRIPT_PROMPT)

  // 缺少 system_prompt 的旧文件导入后回落空串（= 使用默认提示词）。
  const legacyFile = importSkillTemplate(
    { script_agent: { camera_composition: 'wide shot' }, storyboard_agent: {} },
    current,
  )
  assert.ok(legacyFile)
  assert.equal(legacyFile!.script_agent.system_prompt, '')
  assert.equal(legacyFile!.storyboard_agent.system_prompt, '')
  assert.equal(legacyFile!.script_agent.camera_composition, 'wide shot')
})

test('导入包裹结构 { template: ... } 与非法文件', () => {
  const current = templateWithPrompts('', '')
  const wrapped = importSkillTemplate({ template: { script_agent: { system_prompt: '包裹结构' } } }, current)
  assert.ok(wrapped)
  assert.equal(wrapped!.script_agent.system_prompt, '包裹结构')

  assert.equal(importSkillTemplate({ foo: 1 }, current), null)
  assert.equal(importSkillTemplate(null, current), null)
})

test('另存为副本保留 system_prompt，仅替换 id 与名称', () => {
  const source = templateWithPrompts(CUSTOM_SCRIPT_PROMPT, CUSTOM_STORYBOARD_PROMPT)
  const copy = templateCopyPayload(source)
  assert.equal(copy.script_agent.system_prompt, CUSTOM_SCRIPT_PROMPT)
  assert.equal(copy.storyboard_agent.system_prompt, CUSTOM_STORYBOARD_PROMPT)
  assert.notEqual(copy.id, source.id)
  assert.match(copy.name, /副本$/)
})

test('重置语义：默认配置的 system_prompt 为空串表示使用后端默认提示词', () => {
  // DEFAULT_AGENT_CONFIG 是「重置为默认值」的目标形态：空串 + 其余默认开关。
  assert.equal(DEFAULT_AGENT_CONFIG.system_prompt, '')
  assert.equal(SYSTEM_PROMPT_MAX_LENGTH, 20000)
  assert.equal(DEFAULT_TEMPLATE.script_agent.system_prompt, '')
  assert.equal(DEFAULT_TEMPLATE.storyboard_agent.system_prompt, '')
})
