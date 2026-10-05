// 组件测试专用的 ESM loader 钩子：
// - Node 原生 type-stripping 只支持 .ts/.mts，遇到 JSX（.tsx）会直接报错；
// - 组件内部大量使用 Vite 风格的无扩展名相对导入（'./BottomBar'），Node ESM
//   默认不补全扩展名。
// 这里借助 vite 自带的 esbuild（零新增依赖）注册 resolve + load 两个 hook，
// 使 `node --test` 能直接 import 渲染层源码。仅由 test-loader.mjs 注册启用。
import { pathToFileURL } from 'node:url'
import { createRequire } from 'node:module'
import { existsSync, readFileSync } from 'node:fs'
import { cacheBaseFor } from './test-cache-paths.mjs'
import path from 'node:path'

const require = createRequire(import.meta.url)

// pnpm 严格 node_modules 下 esbuild 不在顶层，但它是 vite 的依赖，
// 可以以 vite 的安装目录为基准解析。
const viteDir = path.dirname(require.resolve('vite'))
const esbuildEntry = require.resolve('esbuild', { paths: [viteDir] })
const { transform } = await import(pathToFileURL(esbuildEntry).href)

// 依赖的 ESM 产物（如 antd/es）内部也是无扩展名相对导入，需包含 .js/.jsx。
const RESOLVE_EXTENSIONS = ['.tsx', '.ts', '.mts', '.jsx', '.mjs', '.js']

// antd 生态包（rc-dialog、@ant-design/cssinjs、@emotion/hash 等）的
// package.json main 指向 CJS 构建，Node ESM 的 default 导入拿到的是整个
// module.exports（命名空间对象），组件渲染会得到「Element type is invalid」
// 或「hash is not a function」。Vite 会选 module 字段的 ESM 构建，这里对
// 包根导入保持一致：优先 module 字段，其次 lib/ → es/ 镜像目录重写。
function nearestPackageJson(fileUrl) {
  let dir = path.dirname(new URL(fileUrl).pathname)
  for (let i = 0; i < 6; i += 1) {
    const candidate = path.join(dir, 'package.json')
    if (existsSync(candidate)) return candidate
    const parent = path.dirname(dir)
    if (parent === dir) return null
    dir = parent
  }
  return null
}

async function preferEsmEntry(specifier, context, nextResolve) {
  const resolved = await nextResolve(specifier, context)
  const url = resolved.url
  if (typeof url !== 'string' || !url.startsWith('file:')) return resolved

  const pkgJsonPath = nearestPackageJson(url)
  if (!pkgJsonPath) return resolved
  const pkg = JSON.parse(readFileSync(pkgJsonPath, 'utf8'))
  const pkgDir = path.dirname(pkgJsonPath)

  if (typeof pkg.module === 'string') {
    const esmEntry = path.resolve(pkgDir, pkg.module)
    if (esmEntry.startsWith(pkgDir) && existsSync(esmEntry)) {
      return { url: pathToFileURL(esmEntry).href, format: 'module', shortCircuit: true }
    }
  }

  if (url.includes('/lib/')) {
    try {
      const twin = await nextResolve(url.replace('/lib/', '/es/'), context)
      if (twin?.url) return twin
    } catch {
      // 没有 es/ 镜像则保留原解析结果
    }
  }
  return resolved
}

// 测试期把 'zustand' 的 React 绑定换成 SSR 实时快照版本（store 语义不变），
// 见 zustandSsrBinding.mjs 顶部注释。必须先于包根 ESM 偏好处理。
const ZUSTAND_SSR_BINDING_URL = pathToFileURL(
  path.resolve(import.meta.dirname, '../src/renderer/test-support/zustandSsrBinding.mjs'),
).href

export async function resolve(specifier, context, nextResolve) {
  if (specifier === 'zustand' && context.parentURL !== ZUSTAND_SSR_BINDING_URL) {
    return { url: ZUSTAND_SSR_BINDING_URL, format: 'module', shortCircuit: true }
  }
  // 仅对「包根」导入做 ESM 构建偏好；子路径（rc-util/es/...）本身已明确。
  const parts = specifier.split('/')
  const isBareRoot =
    !specifier.startsWith('.') &&
    !specifier.startsWith('/') &&
    !specifier.startsWith('file:') &&
    (parts.length === 1 || (specifier.startsWith('@') && parts.length === 2))
  if (isBareRoot) {
    return preferEsmEntry(specifier, context, nextResolve)
  }
  try {
    const resolved = await nextResolve(specifier, context)
    return resolved
  } catch (error) {
    // 依次尝试补全源码扩展名与 index 文件。覆盖三类 Node ESM 不支持、
    // 而 Vite/bundler 支持的导入：项目源码的 './BottomBar'（.tsx）、
    // 依赖产物内部的 './button'（.js）、裸子路径 'rc-util/es/hooks/useLayoutEffect'。
    const candidates = []
    for (const ext of RESOLVE_EXTENSIONS) {
      candidates.push(specifier + ext, `${specifier}/index${ext}`)
    }
    for (const candidate of candidates) {
      try {
        return await nextResolve(candidate, context)
      } catch {
        // 继续尝试下一个候选
      }
    }
    throw error
  }
}

export async function load(url, context, nextLoad) {
  if (!/\.(tsx|jsx)$/.test(url)) return nextLoad(url, context)
  const fileUrl = new URL(url)
  if (fileUrl.protocol !== 'file:') return nextLoad(url, context)

  const sourcePath = fileUrl.pathname
  // 优先读预转换缓存（build-test-cache.mjs 产物）：多测试子进程并发现场
  // 转换会触发 esbuild 服务竞态，导致文件被 runner 静默跳过。
  try {
    const cacheBase = cacheBaseFor(sourcePath)
    const cacheFile = `${cacheBase}.mjs`
    const mtimeFile = `${cacheBase}.mtime`
    const sourceMtime = String(statSync(sourcePath).mtimeMs)
    if (existsSync(cacheFile) && existsSync(mtimeFile) && readFileSync(mtimeFile, 'utf8') === sourceMtime) {
      return { format: 'module', source: readFileSync(cacheFile, 'utf8'), shortCircuit: true }
    }
  } catch {
    // 缓存读取失败则回落到现场转换
  }

  const source = readFileSync(fileUrl, 'utf8')
  const result = await transform(source, {
    loader: 'tsx',
    jsx: 'automatic',
    format: 'esm',
    sourcemap: 'inline',
    sourcefile: sourcePath,
  })
  return { format: 'module', source: result.code, shortCircuit: true }
}
