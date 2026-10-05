// 测试转换缓存的路径工具（无副作用，供 hooks 与构建脚本共用）。
import path from 'node:path'

const clientRoot = path.resolve(import.meta.dirname, '..')

export const CACHE_DIR = path.join(clientRoot, 'node_modules', '.cache', 'comic-agent-test-loader')

/**
 * 缓存布局：<相对路径>.mjs 存转换产物，伴生 <相对路径>.mtime 存源文件
 * mtime 字符串；两者齐全且 mtime 一致才算命中。
 */
export function cacheBaseFor(sourcePath) {
  const rel = path.relative(clientRoot, sourcePath)
  return path.join(CACHE_DIR, rel)
}
