// 运行时缓存标记的单元测试:覆盖命中、锁文件变化、PBS 资产变化、
// 标记版本变化以及旧版标记(缺少新字段)等分支。
//
// 用例只操作内存中的对象,不落盘,因此不会在 git 工作区留下构建产物。
import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  MISSING_REQUIREMENTS_LOCK_SHA256,
  RUNTIME_BUILD_SCRIPT_VERSION,
  RUNTIME_MARKER_SCHEMA_VERSION,
  createRuntimeMarker,
  markerMatches,
} from './runtimeMarker.mjs'

const ASSET = 'cpython-3.11.16+20260901-aarch64-apple-darwin-install_only_stripped.tar.gz'
const ASSET_SHA256 = '1'.repeat(64)
const LOCK_SHA256 = '2'.repeat(64)

const expected = createRuntimeMarker({
  asset: ASSET,
  assetSha256: ASSET_SHA256,
  requirementsLockSha256: LOCK_SHA256,
})

test('标记完全一致时命中缓存(复用已安装的运行时)', () => {
  assert.equal(markerMatches({ ...expected }, expected), true)
  // 写入磁盘再读回来的是普通 JSON 对象,同样必须命中。
  assert.equal(markerMatches(JSON.parse(JSON.stringify(expected)), expected), true)
})

test('requirements.lock 的 sha256 变化时缓存失效', () => {
  assert.equal(
    markerMatches({ ...expected, requirementsLockSha256: '3'.repeat(64) }, expected),
    false,
    '锁文件被重新编译后必须重装依赖',
  )
})

test('PBS 资产名或资产 sha256 变化时缓存失效', () => {
  assert.equal(
    markerMatches(
      {
        ...expected,
        asset: 'cpython-3.11.16+20260901-x86_64-apple-darwin-install_only_stripped.tar.gz',
      },
      expected,
    ),
    false,
  )
  assert.equal(markerMatches({ ...expected, assetSha256: '4'.repeat(64) }, expected), false)
})

test('标记 schema 版本或构建脚本版本变化时缓存失效', () => {
  assert.equal(markerMatches({ ...expected, schemaVersion: RUNTIME_MARKER_SCHEMA_VERSION + 1 }, expected), false)
  assert.equal(markerMatches({ ...expected, scriptVersion: RUNTIME_BUILD_SCRIPT_VERSION + 1 }, expected), false)
})

test('旧版标记缺少新增字段时一律视为过期', () => {
  // 修复前的标记格式:只有资产名/sha256 和锁文件路径,没有 schemaVersion 等字段。
  const legacy = {
    asset: ASSET,
    sha256: ASSET_SHA256,
    lockFile: 'server/requirements.lock',
  }
  assert.equal(markerMatches(legacy, expected), false, '旧标记缺少新字段,绝不能复用')
  assert.equal(markerMatches({ asset: ASSET }, expected), false, '只记录资产名的标记也必须失效')
  assert.equal(markerMatches({}, expected), false, '空对象必须失效')
})

test('标记损坏或类型异常时按过期处理而不是抛错', () => {
  for (const broken of [null, undefined, 'not json', 42, [], [expected]]) {
    assert.equal(markerMatches(broken, expected), false, `损坏标记 ${JSON.stringify(broken)} 必须失效`)
  }
})

test('requirements.lock 缺失时使用哨兵值,不会抛出异常', () => {
  for (const missing of [undefined, null, '']) {
    const marker = createRuntimeMarker({
      asset: ASSET,
      assetSha256: ASSET_SHA256,
      requirementsLockSha256: missing,
    })
    assert.equal(marker.requirementsLockSha256, MISSING_REQUIREMENTS_LOCK_SHA256)
    assert.equal(markerMatches(marker, expected), false, '哨兵值不等于真实锁文件哈希,不能命中')
  }
  assert.equal(expected.requirementsLockSha256, LOCK_SHA256, '有锁文件时必须记录真实 sha256')
})
