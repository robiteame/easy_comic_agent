/** 系统设置页的小型表单控件。 */

import { ReloadOutlined } from '@ant-design/icons'
import Button from 'antd/es/button'
import Input from 'antd/es/input'
import InputNumber from 'antd/es/input-number'
import Slider from 'antd/es/slider'
import Switch from 'antd/es/switch'
import { SYSTEM_PROMPT_MAX_LENGTH } from './skillTemplateModel'

const { TextArea } = Input

// 系统提示词编辑框：多行输入，留空 = 使用后端内置默认提示词；JSON 输出
// 契约由后端固定拼装，用户提示词无法删除结构约束。计数按去首尾空白后的长度。
export function SystemPromptField({
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

export function ToggleRow({
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

export function WeightField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (value: number) => void
}) {
  return (
    <div className="skill-weight-row">
      <span>{label}</span>
      <Slider min={0} max={1} step={0.05} value={value} onChange={onChange} />
      <InputNumber min={0} max={1} step={0.05} value={value} onChange={(next) => onChange(Number(next || 0))} />
    </div>
  )
}
