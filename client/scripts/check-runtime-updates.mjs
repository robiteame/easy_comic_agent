#!/usr/bin/env node
// Monthly runtime security-version check.
//
// pnpm audit does not cover vulnerabilities in Electron's bundled Chromium,
// and the desktop bundle also ships a compiled FFmpeg (parses untrusted
// media) and a python-build-standalone runtime. This script compares the
// pinned versions in the repo against upstream support data:
//
//   Electron  client/package.json           npm dist-tags (latest 3 majors supported)
//   FFmpeg    build-python-runtime.mjs      endoflife.date API
//   PBS       build-python-runtime.mjs      GitHub latest release
//
// Exit code 1 when a pinned runtime is outside its upstream support window
// (`.github/workflows/runtime-update-check.yml` turns that into a tracking
// issue). Informational deltas exit 0.
//
// Usage: node check-runtime-updates.mjs [--report <file>]

import { readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const clientRoot = path.resolve(__dirname, '..')
const _repoRoot = path.resolve(clientRoot, '..')

const args = process.argv.slice(2)
const reportIndex = args.indexOf('--report')
const reportFile = reportIndex >= 0 ? args[reportIndex + 1] : null

// Electron keeps the latest three major versions on security support.
const ELECTRON_SUPPORTED_WINDOW = 3

function readPinned() {
  const pkg = JSON.parse(readFileSync(path.join(clientRoot, 'package.json'), 'utf8'))
  const electronRange = pkg.devDependencies?.electron ?? ''
  const electronMatch = electronRange.match(/(\d+)\.(\d+)\.(\d+)/)

  const runtimeScript = readFileSync(path.join(clientRoot, 'scripts', 'build-python-runtime.mjs'), 'utf8')
  const ffmpegVersion = runtimeScript.match(/const FFMPEG_VERSION = '([^']+)'/)?.[1]
  const pbsTag = runtimeScript.match(/const PBS_RELEASE_TAG = '([^']+)'/)?.[1]
  const pbsPython = runtimeScript.match(/const PBS_PYTHON_VERSION = '([^']+)'/)?.[1]

  return {
    electron: electronMatch ? `${electronMatch[1]}.${electronMatch[2]}.${electronMatch[3]}` : null,
    electronMajor: electronMatch ? Number(electronMatch[1]) : null,
    ffmpeg: ffmpegVersion ?? null,
    pbsTag: pbsTag ?? null,
    pbsPython: pbsPython ?? null,
  }
}

async function fetchJson(url, headers = {}) {
  const response = await fetch(url, { headers: { accept: 'application/json', ...headers } })
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`)
  return response.json()
}

const checks = []

function record(item) {
  checks.push(item)
}

async function checkElectron(pinned) {
  const distTags = await fetchJson('https://registry.npmjs.org/-/package/electron/dist-tags')
  const latest = distTags.latest
  if (!latest || !pinned.electron) {
    record({ name: 'Electron', pinned: pinned.electron ?? '?', latest: '?', status: 'unknown', note: '无法解析版本' })
    return
  }
  const latestMajor = Number(latest.split('.')[0])
  const majorGap = latestMajor - pinned.electronMajor
  // pnpm audit 不检查 Electron 内嵌的 Chromium 漏洞；Electron 只对最新 3 个
  // 主版本发布安全补丁，掉出窗口即不再有安全更新。
  if (majorGap >= ELECTRON_SUPPORTED_WINDOW) {
    record({
      name: 'Electron',
      pinned: pinned.electron,
      latest,
      status: 'out-of-support',
      note: `落后 ${majorGap} 个主版本，已脱离 Electron 安全维护窗口（最新 3 个主版本）`,
    })
  } else if (majorGap > 0) {
    record({
      name: 'Electron',
      pinned: pinned.electron,
      latest,
      status: 'info',
      note: `落后 ${majorGap} 个主版本，仍在安全维护窗口内`,
    })
  } else {
    record({ name: 'Electron', pinned: pinned.electron, latest, status: 'ok', note: '' })
  }
}

async function checkFfmpeg(pinned) {
  const cycles = await fetchJson('https://endoflife.date/api/ffmpeg.json')
  if (!pinned.ffmpeg) {
    record({ name: 'FFmpeg', pinned: '?', latest: '?', status: 'unknown', note: '无法解析钉定版本' })
    return
  }
  const pinnedMajor = pinned.ffmpeg.split('.')[0]
  // endoflife.date keys ffmpeg cycles as "9.0" / "8.1"; match on the major.
  const entry = cycles.find((cycle) => String(cycle.cycle).split('.')[0] === pinnedMajor)
  const newestCycle = cycles
    .map((cycle) => Number(cycle.cycle))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => b - a)[0]
  // endoflife.date returns eol as true/false or an ISO date string.
  const isEol = (value) => {
    if (value === true) return true
    if (typeof value === 'string') {
      const date = new Date(value)
      return !Number.isNaN(date.getTime()) && date < new Date()
    }
    return false
  }
  const branchEol = entry ? isEol(entry.eol) : true
  if (!entry || branchEol) {
    record({
      name: 'FFmpeg',
      pinned: pinned.ffmpeg,
      latest: entry?.latest ?? String(newestCycle),
      status: 'out-of-support',
      note: entry?.eol
        ? `上游分支 ${pinnedMajor}.x 已于 ${entry.eol} 结束维护`
        : `上游无 ${pinnedMajor}.x 的维护数据，当前最新分支为 ${newestCycle}.x`,
    })
  } else if (entry.latest !== pinned.ffmpeg) {
    record({
      name: 'FFmpeg',
      pinned: pinned.ffmpeg,
      latest: entry.latest,
      status: 'update-available',
      note: `${pinnedMajor}.x 分支有更新的补丁版本`,
    })
  } else {
    record({ name: 'FFmpeg', pinned: pinned.ffmpeg, latest: entry.latest, status: 'ok', note: '' })
  }
}

async function checkPbs(pinned) {
  const release = await fetchJson('https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest')
  const latest = release.tag_name
  if (!pinned.pbsTag) {
    record({
      name: 'Python runtime (PBS)',
      pinned: '?',
      latest: latest ?? '?',
      status: 'unknown',
      note: '无法解析钉定 tag',
    })
    return
  }
  if (pinned.pbsTag === latest) {
    record({ name: 'Python runtime (PBS)', pinned: pinned.pbsTag, latest, status: 'ok', note: '' })
  } else {
    record({
      name: 'Python runtime (PBS)',
      pinned: pinned.pbsTag,
      latest,
      status: 'update-available',
      note: `CPython ${pbsPythonOf(pinned)}；升级时同步更新 PBS_ASSETS 的 sha256`,
    })
  }
}

function pbsPythonOf(pinned) {
  return pinned.pbsPython ?? '3.11.x'
}

const pinned = readPinned()
for (const [label, run] of [
  ['Electron', () => checkElectron(pinned)],
  ['FFmpeg', () => checkFfmpeg(pinned)],
  ['PBS', () => checkPbs(pinned)],
]) {
  try {
    await run()
  } catch (error) {
    record({ name: label, pinned: '?', latest: '?', status: 'unknown', note: `查询失败: ${error.message}` })
  }
}

const breaking = checks.filter((item) => item.status === 'out-of-support')
const _available = checks.filter((item) => item.status === 'update-available')

const statusLabel = { ok: '✅', info: 'ℹ️', 'update-available': '⬆️', 'out-of-support': '❌', unknown: '⚠️' }
const lines = [
  '# 运行时安全版本检查',
  '',
  `检查时间: ${new Date().toISOString()}`,
  '',
  '| 组件 | 当前钉定 | 上游最新 | 状态 | 说明 |',
  '| --- | --- | --- | --- | --- |',
  ...checks.map(
    (item) =>
      `| ${item.name} | ${item.pinned} | ${item.latest} | ${statusLabel[item.status]} ${item.status} | ${item.note} |`,
  ),
  '',
]

if (breaking.length > 0) {
  lines.push(
    '## 需要立即处理',
    '',
    ...breaking.map((item) => `- **${item.name}** ${item.pinned}: ${item.note}`),
    '',
    '请升级钉定版本并重建 lock 文件 / 运行时（`pnpm install`、`pnpm --dir client run build:runtime`）。',
    '',
  )
}

const report = lines.join('\n')
console.log(report)
if (reportFile) writeFileSync(reportFile, `${report}\n`)
if (process.env.GITHUB_STEP_SUMMARY) {
  const { appendFileSync } = await import('node:fs')
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${report}\n`)
}

if (breaking.length > 0) {
  console.error(`[runtime-check] ${breaking.length} 个运行时已脱离上游安全维护`)
  process.exit(1)
}
console.log('[runtime-check] 无脱离安全维护窗口的运行时')
