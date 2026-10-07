import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const root = path.resolve(import.meta.dirname, '..')
const maxLines = 1200
const sourceExtensions = new Set(['.css', '.mjs', '.mts', '.py', '.ts', '.tsx'])
const skippedDirectories = new Set([
  '.git',
  '.pytest_cache',
  '.ruff_cache',
  '.venv',
  '__pycache__',
  'build',
  'dist',
  'dist-electron',
  'node_modules',
  'release',
])

// 存量巨型文件只允许缩短；基线取 P1 落地时的实测行数。
const legacyLineLimits = new Map([
  ['client/src/renderer/components/MainWorkspace.tsx', 2542],
  ['client/src/renderer/components/RightSidebar.tsx', 1265],
  ['client/src/renderer/components/SystemSettingsPage.tsx', 1448],
  ['client/src/renderer/styles/global.css', 7745],
  ['server/agent/contracts.py', 1259],
  ['server/agent/critic.py', 1540],
  ['server/agent/graph.py', 3641],
  ['server/api/routes/shot.py', 3181],
  ['server/services/budget_service.py', 1213],
  ['server/services/story_timing.py', 1913],
  ['server/tests/test_regressions.py', 1388],
])

function walk(directory, files = []) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (!skippedDirectories.has(entry.name)) walk(path.join(directory, entry.name), files)
      continue
    }
    const extension = path.extname(entry.name)
    if (entry.isFile() && sourceExtensions.has(extension)) files.push(path.join(directory, entry.name))
  }
  return files
}

function lineCount(file) {
  const content = readFileSync(file, 'utf8')
  return content.split('\n').length - 1
}

const failures = []
for (const file of [...walk(path.join(root, 'client')), ...walk(path.join(root, 'server'))]) {
  const relativePath = path.relative(root, file).split(path.sep).join('/')
  const lines = lineCount(file)
  const legacyLimit = legacyLineLimits.get(relativePath)
  if (legacyLimit === undefined && lines > maxLines) {
    failures.push(`${relativePath}: ${lines} 行，超过新文件上限 ${maxLines}`)
  } else if (legacyLimit !== undefined && lines > legacyLimit) {
    failures.push(`${relativePath}: ${lines} 行，超过存量白名单基线 ${legacyLimit}（只减不增）`)
  }
}

if (failures.length > 0) {
  console.error('文件行数门禁失败：')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`文件行数门禁通过：新文件不超过 ${maxLines} 行；${legacyLineLimits.size} 个存量文件保持只减不增。`)
