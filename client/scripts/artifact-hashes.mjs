#!/usr/bin/env node
// Compute sha256 checksums for the built desktop installers. Runs on every
// release runner OS, so it uses Node's crypto instead of shasum/sha256sum,
// which do not exist uniformly across macOS and Windows shells.
//
// Usage: node artifact-hashes.mjs <release-dir> --output <name>

import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import path from 'node:path'

const args = process.argv.slice(2)
const outputIndex = args.indexOf('--output')
const outputName = outputIndex >= 0 ? args[outputIndex + 1] : 'SHA256SUMS.txt'
// --output 不存在时（outputIndex=-1）必须保留全部位置参数；
// 此前 -1 + 1 = 0 会把 release-dir 本身误当作输出名丢弃。
const positional =
  outputIndex >= 0 ? args.filter((_, index) => index !== outputIndex && index !== outputIndex + 1) : args
const releaseDir = positional[0]
if (!releaseDir) {
  console.error('用法: node artifact-hashes.mjs <release-dir> --output <name>')
  process.exit(1)
}

const HASHABLE = /\.(dmg|zip|exe|blockmap)$/i
const entries = readdirSync(releaseDir)
  .filter((name) => HASHABLE.test(name) && name !== outputName)
  .filter((name) => statSync(path.join(releaseDir, name)).isFile())
  .sort()

if (entries.length === 0) {
  console.error(`[artifact-hashes] ${releaseDir} 中没有可校验的安装包`)
  process.exit(1)
}

const lines = entries.map((name) => {
  const digest = createHash('sha256')
    .update(readFileSync(path.join(releaseDir, name)))
    .digest('hex')
  return `${digest}  ${name}`
})

const outputPath = path.join(releaseDir, outputName)
writeFileSync(outputPath, `${lines.join('\n')}\n`)
console.log(`[artifact-hashes] ${outputPath}`)
for (const line of lines) console.log(line)
