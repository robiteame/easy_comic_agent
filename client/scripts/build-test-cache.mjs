// 测试前的 .tsx 预转换缓存：单进程把全部 JSX 源码转成纯 JS 缓存。
//
// 背景：node --test 的每个测试文件跑在独立子进程，组件测试通过 loader
// 现场调 esbuild 转换 .tsx 时，多进程并发冷启动 esbuild 服务存在竞态，
// 会导致部分测试文件被 runner 静默跳过。这里在测试启动前用单进程完成
// 全部转换，load hook 之后只读缓存，不再触碰 esbuild。
//
// 用法：node scripts/build-test-cache.mjs（由 npm test 自动前置）
import { pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { cacheBaseFor } from './test-cache-paths.mjs'

const require = createRequire(import.meta.url)
const clientRoot = path.resolve(import.meta.dirname, '..')

const viteDir = path.dirname(require.resolve('vite', { paths: [clientRoot] }))
const esbuildEntry = require.resolve('esbuild', { paths: [viteDir] })
const { transform } = await import(pathToFileURL(esbuildEntry).href)

function walk(dir) {
  const out = []
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name)
    const stat = statSync(full)
    if (stat.isDirectory()) out.push(...walk(full))
    else if (/\.(tsx|jsx)$/.test(name)) out.push(full)
  }
  return out
}

let converted = 0
let reused = 0
for (const file of walk(path.join(clientRoot, 'src'))) {
  const cacheBase = cacheBaseFor(file)
  const cacheFile = `${cacheBase}.mjs`
  const mtimeFile = `${cacheBase}.mtime`
  const sourceMtime = String(statSync(file).mtimeMs)
  if (
    statSync(cacheFile, { throwIfNoEntry: false }) &&
    statSync(mtimeFile, { throwIfNoEntry: false }) &&
    readFileSync(mtimeFile, 'utf8') === sourceMtime
  ) {
    reused += 1
    continue
  }
  const result = await transform(readFileSync(file, 'utf8'), {
    loader: 'tsx',
    jsx: 'automatic',
    format: 'esm',
    sourcemap: 'inline',
    sourcefile: file,
  })
  mkdirSync(path.dirname(cacheBase), { recursive: true })
  writeFileSync(cacheFile, result.code)
  writeFileSync(mtimeFile, sourceMtime)
  converted += 1
}
console.log(`[test-cache] 转换 ${converted} 个 JSX 文件，复用缓存 ${reused} 个`)
