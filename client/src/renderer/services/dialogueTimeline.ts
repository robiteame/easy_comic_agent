export interface EditableDialogueLine {
  speaker?: string
  line: string
  emotion?: string
  action?: string
  start_ms?: number | null
  end_ms?: number | null
}

export type EditableDialogue = string | EditableDialogueLine[]

const EMOTIONS = new Set(['neutral', 'happy', 'shy', 'sad', 'angry', 'surprised'])

function numberOrNull(value: string): number | null {
  const trimmed = value.trim()
  if (!trimmed || trimmed === 'auto') return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : null
}

export function formatDialogueForEditor(value: EditableDialogue | null | undefined): string {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''
  return value
    .map((item) => {
      const timing = item.start_ms == null || item.end_ms == null ? 'auto' : `${item.start_ms}-${item.end_ms}`
      return `${item.speaker || ''} | ${item.emotion || 'neutral'} | ${timing} | ${item.line || ''}`
    })
    .join('\n')
}

export function parseDialogueFromEditor(value: string, _fallbackSpeaker = ''): EditableDialogue {
  const text = value.trim()
  if (!text) return []
  const rows = text
    .split(/\r?\n/)
    .map((row) => row.trim())
    .filter(Boolean)
  const structured = rows.every((row) => row.includes('|'))
  if (!structured) return value

  return rows
    .map((row) => {
      const [speaker = '', emotion = '', timing = '', ...lineParts] = row.split('|')
      const line = lineParts.join('|').trim()
      const match = timing.trim().match(/^(\d+|auto)\s*-\s*(\d+|auto)$/)
      return {
        speaker: speaker.trim(),
        line,
        emotion: EMOTIONS.has(emotion.trim()) ? emotion.trim() : 'neutral',
        start_ms: match ? numberOrNull(match[1]) : null,
        end_ms: match ? numberOrNull(match[2]) : null,
      }
    })
    .filter((item) => item.line)
}
