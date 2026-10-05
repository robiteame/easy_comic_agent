// 运行时缓存标记(pure 逻辑,无 fs / 无副作用)。
//
// build-python-runtime.mjs 用它判断 client/build/python 里的运行时能否直接复用:
// 标记同时记录 PBS 资产与 server/requirements.lock 的 sha256,任何一项变化
// 都意味着 site-packages 已经过期,必须重装依赖而不是跳过。
//
// 这里只做纯函数计算,便于在 node:test 用例中直接覆盖命中/失效分支;
// 读文件、算 sha256、写标记等有副作用的逻辑留在构建脚本里。

// 标记结构变更时递增(例如新增/重命名字段)。旧标记缺少新字段时会被
// markerMatches 判为过期,因此升级只会多花一次构建时间,不会复用错缓存。
export const RUNTIME_MARKER_SCHEMA_VERSION = 2

// 构建逻辑变更时递增(依赖安装方式、瘦身步骤、relocation 自检等),
// 强制重建已存在的运行时。
export const RUNTIME_BUILD_SCRIPT_VERSION = 2

// server/requirements.lock 缺失时的哨兵值:标记计算不能因为读不到锁文件而
// 抛异常(真正安装依赖时脚本会给出更明确的报错)。
export const MISSING_REQUIREMENTS_LOCK_SHA256 = 'missing'

// 必须逐项比对且不允许缺失的字段。
const MARKER_FIELDS = ['schemaVersion', 'scriptVersion', 'asset', 'assetSha256', 'requirementsLockSha256']

// 期望值必须是有效值(非空字符串或整数),否则一律视为不匹配,
// 避免 undefined === undefined 之类的"假命中"。
function isUsableExpectedValue(value) {
  return typeof value === 'string' ? value.length > 0 : Number.isInteger(value)
}

export function createRuntimeMarker({ asset, assetSha256, requirementsLockSha256 }) {
  return {
    schemaVersion: RUNTIME_MARKER_SCHEMA_VERSION,
    scriptVersion: RUNTIME_BUILD_SCRIPT_VERSION,
    asset,
    assetSha256,
    requirementsLockSha256: requirementsLockSha256 || MISSING_REQUIREMENTS_LOCK_SHA256,
  }
}

// 磁盘上的标记(可能来自旧版本构建,也可能已损坏)与本次构建的期望值逐项比对。
// 只要有一项不同、缺失或类型不对,就返回 false(缓存未命中 -> 重装依赖)。
export function markerMatches(cached, expected) {
  if (!cached || typeof cached !== 'object' || Array.isArray(cached)) return false
  if (!expected || typeof expected !== 'object') return false
  return MARKER_FIELDS.every((field) => isUsableExpectedValue(expected[field]) && cached[field] === expected[field])
}
