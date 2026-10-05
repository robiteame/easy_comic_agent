import assert from 'node:assert/strict'
import { test } from 'node:test'
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), 'artifact-hashes.mjs')

function makeReleaseDir() {
  return mkdtempSync(path.join(tmpdir(), 'artifact-hashes-test-'))
}

function run(args) {
  return spawnSync(process.execPath, [scriptPath, ...args], { encoding: 'utf8' })
}

test('对安装包生成 sha256 校验和文件并排除输出文件自身', () => {
  const dir = makeReleaseDir()
  try {
    writeFileSync(path.join(dir, 'ComicAgent-0.2.0.dmg'), 'dmg-bytes')
    writeFileSync(path.join(dir, 'ComicAgent-0.2.0.dmg.blockmap'), 'blockmap-bytes')
    writeFileSync(path.join(dir, 'ComicAgent-Setup.exe'), 'exe-bytes')
    // 非安装包扩展名应被忽略
    writeFileSync(path.join(dir, 'latest.yml'), 'yml-bytes')
    writeFileSync(path.join(dir, 'notes.txt'), 'txt-bytes')

    const result = run([dir])
    assert.equal(result.status, 0, `脚本应成功退出：${result.stderr}`)

    const sums = readFileSync(path.join(dir, 'SHA256SUMS.txt'), 'utf8')
    const lines = sums.trim().split('\n')
    assert.equal(lines.length, 3, '应只包含 3 个安装包文件')
    assert.ok(sums.includes('ComicAgent-0.2.0.dmg'))
    assert.ok(sums.includes('ComicAgent-0.2.0.dmg.blockmap'))
    assert.ok(sums.includes('ComicAgent Setup.exe') || sums.includes('ComicAgent-Setup.exe'))
    assert.ok(!sums.includes('latest.yml'), 'yml 不应计入')
    assert.ok(!sums.includes('notes.txt'), 'txt 不应计入')
    assert.ok(!sums.includes('SHA256SUMS.txt'), '输出文件自身不应计入')

    // 摘要格式：<sha256 两个空格 文件名>，且与实际内容一致
    const expected = createHash('sha256').update('dmg-bytes').digest('hex')
    assert.ok(
      lines.some((line) => line === `${expected}  ComicAgent-0.2.0.dmg`),
      '摘要应与文件内容一致',
    )
    for (const line of lines) {
      assert.match(line, /^[0-9a-f]{64} {2}\S+$/, '每行应是「sha256␣␣文件名」格式')
    }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('自定义 --output 文件名生效', () => {
  const dir = makeReleaseDir()
  try {
    writeFileSync(path.join(dir, 'app.zip'), 'zip-bytes')
    const result = run([dir, '--output', 'CHECKSUMS.txt'])
    assert.equal(result.status, 0, `应成功退出：${result.stderr}`)
    const sums = readFileSync(path.join(dir, 'CHECKSUMS.txt'), 'utf8')
    assert.ok(sums.includes('app.zip'))
    // 默认名不应生成
    assert.throws(() => readFileSync(path.join(dir, 'SHA256SUMS.txt')), '自定义输出名时不应生成默认文件')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('缺失 release-dir 参数时报用法错误退出 1', () => {
  const result = run([])
  assert.equal(result.status, 1)
  assert.ok(result.stderr.includes('用法'), '应输出用法提示')
})

test('目录中没有可校验的安装包时退出 1', () => {
  const dir = makeReleaseDir()
  try {
    writeFileSync(path.join(dir, 'readme.md'), 'no installers here')
    const result = run([dir])
    assert.equal(result.status, 1)
    assert.ok(result.stderr.includes('没有可校验的安装包'), '应提示没有安装包')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('子目录中的安装包不参与校验（只处理顶层文件）', () => {
  const dir = makeReleaseDir()
  try {
    writeFileSync(path.join(dir, 'top.zip'), 'top-bytes')
    mkdirSync(path.join(dir, 'nested'))
    writeFileSync(path.join(dir, 'nested', 'inner.zip'), 'inner-bytes')

    const result = run([dir])
    assert.equal(result.status, 0)
    const sums = readFileSync(path.join(dir, 'SHA256SUMS.txt'), 'utf8')
    assert.ok(sums.includes('top.zip'))
    assert.ok(!sums.includes('inner.zip'), '子目录文件不应计入')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
