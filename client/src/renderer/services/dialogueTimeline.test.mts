import assert from 'node:assert/strict'
import test from 'node:test'
import { formatDialogueForEditor, parseDialogueFromEditor } from './dialogueTimeline.ts'

test('structured dialogue preserves speaker emotion and timing through editor text', () => {
  const source = [
    { speaker: '阿宁', line: '别过来', emotion: 'angry', start_ms: 0, end_ms: 900 },
    { speaker: '小满', line: '好', emotion: 'sad', start_ms: 900, end_ms: 1200 },
  ]
  const restored = parseDialogueFromEditor(formatDialogueForEditor(source), '兜底')
  assert.deepEqual(restored, source)
})

test('legacy plain text remains plain text for backend compatibility', () => {
  assert.equal(parseDialogueFromEditor('只有一句普通台词', '阿宁'), '只有一句普通台词')
})

test('structured rows without speaker stay unattributed instead of borrowing the first character', () => {
  const restored = parseDialogueFromEditor(' | happy | auto | 台词甲\n小满 | sad | auto | 台词乙')
  assert.deepEqual(restored, [
    { speaker: '', line: '台词甲', emotion: 'happy', start_ms: null, end_ms: null },
    { speaker: '小满', line: '台词乙', emotion: 'sad', start_ms: null, end_ms: null },
  ])
})
