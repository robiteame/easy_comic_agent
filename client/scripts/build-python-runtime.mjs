#!/usr/bin/env node
// Builds the relocatable desktop runtime under client/build/:
//
//   build/python/   python-build-standalone (PBS) CPython + production deps
//   build/bin/      pinned static ffmpeg for the current platform
//   build/THIRD-PARTY-NOTICES.md
//
// The whole tree is mapped into the packaged app via electron-builder
// extraResources and must stay self-contained: no venv, no absolute paths,
// no reliance on a system Python or ffmpeg.
//
// All downloads are pinned to an exact version and verified against a
// sha256 constant. Never point these at a rolling "latest" URL.
//
// Usage:
//   node scripts/build-python-runtime.mjs [--clean] [--platform darwin|win32] [--arch arm64|x64]
//
// --platform/--arch override the host triple (electron-builder builds are
// per-runner, so cross-builds are only useful for debugging).

import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  chmodSync,
  copyFileSync,
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { createRuntimeMarker, markerMatches } from './runtimeMarker.mjs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const clientRoot = path.resolve(__dirname, '..')
const serverRoot = path.resolve(clientRoot, '..', 'server')
const buildDir = path.join(clientRoot, 'build')
const cacheDir = path.join(buildDir, 'cache')
const pythonDir = path.join(buildDir, 'python')
const binDir = path.join(buildDir, 'bin')
const runtimeMarker = path.join(pythonDir, '.comicagent-runtime.json')

// ---------------------------------------------------------------------------
// Pinned artifacts
// ---------------------------------------------------------------------------

// server/requirements.lock is compiled for Python 3.11, so stay on 3.11.x.
const PBS_PYTHON_VERSION = '3.11.16'
const PBS_RELEASE_TAG = '20260901'
const PBS_VARIANT = 'install_only_stripped'
const PBS_ASSETS = {
  'darwin-arm64': {
    triple: 'aarch64-apple-darwin',
    sha256: '768f05cf200273bbdda9a5955a5a6892a4b22f2a0b1e4b0a9160f5c7fce86816',
  },
  'darwin-x64': {
    triple: 'x86_64-apple-darwin',
    sha256: '908b381433f78b832c8d64960ced0f85871893cc8779f413f963e0c9e293c258',
  },
  'win32-x64': {
    triple: 'x86_64-pc-windows-msvc',
    sha256: '06cbe479e039f5b9cb5640c286d790074d63f549f92a32d599a3748293bd4510',
  },
}

// ffmpeg: the server encodes with libx264 (services/ffmpeg_service.py), so
// the build must be a GPL flavour bundling x264. The binary parses uploaded
// and remotely downloaded media, so it must come from a maintained FFmpeg
// branch. macOS builds are compiled from the official release tarball plus a
// pinned x264 snapshot (only system frameworks are linked — see otool -L);
// Windows uses gyan.dev's versioned essentials build. All artifacts are
// pinned to exact versions and sha256 constants; never point at rolling
// "latest" URLs.
const FFMPEG_VERSION = '9.0.1'
const FFMPEG_SOURCE_SHA256 = 'cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635'
const FFMPEG_SOURCE_URL = `https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz`

// macOS libx264 from the VideoLAN "stable" branch at a pinned commit.
const X264_COMMIT = 'b35605ace3ddf7c1a5d67a2eb553f034aef41d55'
const X264_SHA256 = 'cd71a7515b0e9a012e1ac9b1f8415bebcaf6fc97d4db32286642ac4c0fbe24f9'
const X264_URL = `https://code.videolan.org/videolan/x264/-/archive/${X264_COMMIT}/x264-${X264_COMMIT}.tar.gz`

// Windows: gyan.dev essentials build (GPL, libx264 included), published via
// the codexffmpeg GitHub release. The sha256 is the release asset digest
// from the GitHub Releases API.
const FFMPEG_WIN_URL = `https://github.com/GyanD/codexffmpeg/releases/download/${FFMPEG_VERSION}/ffmpeg-${FFMPEG_VERSION}-essentials_build.zip`
const FFMPEG_WIN_SHA256 = 'fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9'
const FFMPEG_WIN_ENTRY = `ffmpeg-${FFMPEG_VERSION}-essentials_build/bin/ffmpeg.exe`

function pbsAssetName(asset) {
  return `cpython-${PBS_PYTHON_VERSION}+${PBS_RELEASE_TAG}-${asset.triple}-${PBS_VARIANT}.tar.gz`
}

function pbsUrl(asset) {
  const name = pbsAssetName(asset)
  return `https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE_TAG}/${encodeURIComponent(name)}`
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function log(msg) {
  console.log(`[build-runtime] ${msg}`)
}

function fail(msg) {
  console.error(`[build-runtime] 错误: ${msg}`)
  process.exit(1)
}

function parseArgs(argv) {
  const args = { clean: false, platform: undefined, arch: undefined }
  for (const raw of argv) {
    if (raw === '--clean') args.clean = true
    else if (raw.startsWith('--platform=')) args.platform = raw.slice('--platform='.length)
    else if (raw.startsWith('--arch=')) args.arch = raw.slice('--arch='.length)
    else fail(`未知参数: ${raw}`)
  }
  return args
}

function resolveTargetKey(args) {
  const platform = args.platform ?? process.platform
  const arch = args.arch ?? process.arch
  const key = `${platform}-${arch}`
  if (!(key in PBS_ASSETS)) {
    fail(`不支持的平台组合: ${key}(支持: ${Object.keys(PBS_ASSETS).join(', ')})`)
  }
  return key
}

function isWindows(targetKey) {
  return targetKey.startsWith('win32-')
}

function pythonExecutable(dir, targetKey) {
  return isWindows(targetKey) ? path.join(dir, 'python.exe') : path.join(dir, 'bin', 'python3')
}

async function sha256File(file) {
  const hash = createHash('sha256')
  await pipeline(createReadStream(file), async (source) => {
    for await (const chunk of source) hash.update(chunk)
  })
  return hash.digest('hex')
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function curlAvailable() {
  return run('curl', ['--version'], { stdio: 'pipe' }).status === 0
}

const HAVE_CURL = curlAvailable()

async function download(url, dest) {
  mkdirSync(path.dirname(dest), { recursive: true })
  // curl copes better with flaky links (its retries also cover stalled
  // connects) and ships on macOS, Linux and Windows 10+ alike; keep a
  // fetch fallback for odd environments without it.
  if (HAVE_CURL) {
    const result = run(
      'curl',
      [
        '--location',
        '--fail',
        '--silent',
        '--show-error',
        '--retry',
        '10',
        '--retry-all-errors',
        '--retry-delay',
        '5',
        '--connect-timeout',
        '30',
        '--output',
        dest,
        url,
      ],
      { stdio: 'inherit' },
    )
    if (result.status !== 0) {
      rmSync(dest, { force: true })
      throw new Error(`curl 退出码 ${result.status}`)
    }
    return
  }
  const maxAttempts = 10
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetch(url, { redirect: 'follow' })
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`)
      await pipeline(Readable.fromWeb(response.body), createWriteStream(dest))
      return
    } catch (error) {
      rmSync(dest, { force: true })
      if (attempt === maxAttempts) throw error
      log(`下载失败(${error.message}),${Math.min(attempt * 5, 30)}s 后重试 ${url}`)
      await wait(Math.min(attempt * 5000, 30000))
    }
  }
}

async function fetchVerified(url, dest, expectedSha256) {
  if (existsSync(dest) && (await sha256File(dest)) === expectedSha256) {
    log(`缓存命中: ${path.basename(dest)}`)
    return
  }
  log(`下载 ${url}`)
  await download(url, dest)
  const actual = await sha256File(dest)
  if (actual !== expectedSha256) {
    rmSync(dest, { force: true })
    fail(`sha256 校验失败: ${path.basename(dest)}\n  期望 ${expectedSha256}\n  实际 ${actual}`)
  }
  log(`sha256 校验通过: ${actual}`)
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: options.stdio ?? 'inherit',
    cwd: options.cwd,
    env: options.env ?? process.env,
    shell: false,
  })
  return result
}

function runInterpreter(interpreter, args) {
  return run(interpreter, ['-c', args.join('\n')], { stdio: 'pipe' })
}

function interpreterCheckCode(dirToExpect) {
  // PBS derives sys.prefix from the interpreter's own location, which is what
  // makes the tree relocatable. Assert the resolved prefix matches the tree
  // we are about to ship.
  return [
    'import os, sys',
    'prefix = os.path.realpath(sys.prefix)',
    `expected = os.path.realpath(${JSON.stringify(dirToExpect)})`,
    'assert prefix == expected, f"sys.prefix {prefix} != {expected}"',
    'assert sys.version_info[:2] == (3, 11), sys.version',
    'print("prefix-ok", prefix, sys.version.split()[0])',
  ]
}

// ---------------------------------------------------------------------------
// Python runtime
// ---------------------------------------------------------------------------

function extractPbs(tarball, stagingDir) {
  // bsdtar ships with Windows 10+ (System32) and macOS/Linux.
  const result = run('tar', ['-xzf', tarball, '-C', stagingDir])
  if (result.status !== 0) fail('解压 PBS 失败(tar 退出码非 0)')
}

function copyTreeDereferenced(source, dest) {
  // PBS macOS archives contain relative symlinks (bin/python3 -> python3.11).
  // Dereferencing avoids dangling symlinks if the packaged tree is copied by
  // a tool that does not preserve them. copyFileSync follows symlink sources
  // and preserves the target's permission bits.
  rmSync(dest, { recursive: true, force: true })
  mkdirSync(dest, { recursive: true })
  const entries = readdirSync(source, { withFileTypes: true })
  for (const entry of entries) {
    const from = path.join(source, entry.name)
    const to = path.join(dest, entry.name)
    if (entry.isDirectory()) copyTreeDereferenced(from, to)
    else copyFileSync(from, to)
  }
}

function removeRecursiveIfExist(dir) {
  if (existsSync(dir)) rmSync(dir, { recursive: true, force: true })
}

function slimPythonRuntime(dir, targetKey) {
  // Drop bytecode caches and the stdlib test suite; neither is ever imported
  // at runtime and together they are roughly 60 MB.
  const removed = { pycache: 0, testDirs: 0 }
  const walk = (current) => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name)
      if (entry.isDirectory()) {
        if (entry.name === '__pycache__') {
          removeRecursiveIfExist(full)
          removed.pycache += 1
          continue
        }
        if (entry.name === 'test' && current === path.join(dir, 'lib', 'python3.11')) {
          removeRecursiveIfExist(full)
          removed.testDirs += 1
          continue
        }
        walk(full)
      } else if (entry.isFile() && /\.py[co]$/.test(entry.name)) {
        rmSync(full, { force: true })
      }
    }
  }
  walk(dir)
  const windowsLibTest = path.join(dir, 'Lib', 'test')
  if (isWindows(targetKey) && existsSync(windowsLibTest)) {
    removeRecursiveIfExist(windowsLibTest)
    removed.testDirs += 1
  }
  log(`瘦身完成: 移除 ${removed.pycache} 个 __pycache__、${removed.testDirs} 个 stdlib test 目录`)
}

function installDependencies(interpreter) {
  const lockFile = path.join(serverRoot, 'requirements.lock')
  if (!existsSync(lockFile)) fail(`未找到 ${lockFile}`)

  // uv installs straight into the PBS prefix (no venv, no pyvenv.cfg), which
  // keeps the tree relocatable.
  const uvAvailable = run('uv', ['--version'], { stdio: 'pipe' }).status === 0
  if (uvAvailable) {
    log('使用 uv 安装生产依赖')
    const result = run('uv', ['pip', 'install', '--python', interpreter, '--require-hashes', '-r', lockFile])
    if (result.status !== 0) fail('uv pip install 失败')
    return
  }

  log('uv 不可用,回退 pip(ensurepip)')
  let result = run(interpreter, ['-m', 'ensurepip', '--upgrade'])
  if (result.status !== 0) fail('ensurepip 失败')
  result = run(interpreter, ['-m', 'pip', 'install', '--no-warn-script-location', '--require-hashes', '-r', lockFile])
  if (result.status !== 0) fail('pip install 失败')
}

async function requirementsLockFingerprint() {
  // 锁文件决定了 site-packages 的内容,必须进入缓存标记:否则改了
  // requirements.lock 之后仍会命中旧标记、跳过重装。文件缺失时不抛错,
  // 交给 createRuntimeMarker 记录哨兵值(后面 installDependencies 会明确报错)。
  const lockFile = path.join(serverRoot, 'requirements.lock')
  if (!existsSync(lockFile)) {
    log('未找到 server/requirements.lock,依赖指纹记为占位值')
    return ''
  }
  return sha256File(lockFile)
}

async function buildPythonRuntime(targetKey) {
  const asset = PBS_ASSETS[targetKey]
  const assetName = pbsAssetName(asset)
  const marker = createRuntimeMarker({
    asset: assetName,
    assetSha256: asset.sha256,
    requirementsLockSha256: await requirementsLockFingerprint(),
  })
  if (existsSync(runtimeMarker)) {
    try {
      const current = JSON.parse(readFileSync(runtimeMarker, 'utf8'))
      if (markerMatches(current, marker)) {
        const check = runInterpreter(pythonExecutable(pythonDir, targetKey), interpreterCheckCode(pythonDir))
        if (check.status === 0) {
          log(`python 运行时已是最新,跳过 (${assetName})`)
          return
        }
        log('现有 python 运行时自检失败,重新构建')
      } else {
        // 资产、锁文件、标记版本任一变化都会走到这里;旧版本标记缺少新增
        // 字段时同样判为过期,必须重装依赖而不是复用。
        log('运行时标记已过期(资产或依赖锁变化),重新构建')
      }
    } catch {
      log('运行时标记损坏,重新构建')
    }
  }

  const tarball = path.join(cacheDir, assetName)
  await fetchVerified(pbsUrl(asset), tarball, asset.sha256)

  log(`解压 ${assetName} -> ${pythonDir}`)
  const stagingDir = path.join(buildDir, 'python-extract')
  removeRecursiveIfExist(stagingDir)
  removeRecursiveIfExist(pythonDir)
  mkdirSync(stagingDir, { recursive: true })
  extractPbs(tarball, stagingDir)
  // The archive contains a single top-level "python/" directory.
  const extractedRoot = path.join(stagingDir, 'python')
  if (!existsSync(extractedRoot)) fail('PBS 包布局异常: 未找到顶层 python/ 目录')
  copyTreeDereferenced(extractedRoot, pythonDir)
  removeRecursiveIfExist(stagingDir)

  const interpreter = pythonExecutable(pythonDir, targetKey)
  if (!existsSync(interpreter)) fail(`捆绑解释器不存在: ${interpreter}`)

  const prefixCheck = runInterpreter(interpreter, interpreterCheckCode(pythonDir))
  if (prefixCheck.status !== 0) {
    fail(`解释器自检失败: ${String(prefixCheck.stderr)}`)
  }
  log(`解释器自检通过: ${String(prefixCheck.stdout).trim()}`)

  installDependencies(interpreter)
  slimPythonRuntime(pythonDir, targetKey)

  // Relocation self-check: move the fully installed tree and confirm the
  // prefix still follows the interpreter. rename is cheap on the same volume.
  const relocated = path.join(buildDir, 'python-relocated')
  removeRecursiveIfExist(relocated)
  renameSync(pythonDir, relocated)
  try {
    const relocatedCheck = runInterpreter(pythonExecutable(relocated, targetKey), interpreterCheckCode(relocated))
    if (relocatedCheck.status !== 0) {
      fail(`relocation 自检失败: ${String(relocatedCheck.stderr)}`)
    }
    log(`relocation 自检通过: ${String(relocatedCheck.stdout).trim()}`)
  } finally {
    renameSync(relocated, pythonDir)
  }

  writeFileSync(runtimeMarker, `${JSON.stringify(marker, null, 2)}\n`)
}

// ---------------------------------------------------------------------------
// ffmpeg
// ---------------------------------------------------------------------------

function ffmpegMarker(targetKey) {
  return {
    version: FFMPEG_VERSION,
    x264Commit: isWindows(targetKey) ? null : X264_COMMIT,
  }
}

function verifyFfmpegBinary(target, _targetKey) {
  const version = run(target, ['-version'], { stdio: 'pipe' })
  if (version.status !== 0) fail(`捆绑 ffmpeg 无法执行: ${target}`)
  const banner = String(version.stdout)
  if (!banner.includes(`ffmpeg version ${FFMPEG_VERSION}`)) {
    fail(`捆绑 ffmpeg 版本异常，期望 ${FFMPEG_VERSION}:\n${banner.split('\n')[0]}`)
  }
  if (!/--enable-libx264/.test(banner)) {
    // 服务端用 libx264 编码（services/ffmpeg_service.py），缺失时会以
    // "Unknown encoder" 失败。
    fail('捆绑 ffmpeg 缺少 libx264 编码器（GPL 构建才包含）')
  }
  log(`ffmpeg 自检通过: ${banner.split('\n')[0]}`)
}

async function buildFfmpegFromSource(buildRoot) {
  const ffmpegSrc = path.join(buildRoot, `ffmpeg-${FFMPEG_VERSION}`)
  const x264Src = path.join(buildRoot, `x264-${X264_COMMIT}`)
  const prefix = path.join(buildRoot, 'prefix')

  // x86_64 assembly (x264 + ffmpeg) is assembled with nasm; arm64 uses
  // intrinsics and the compiler's integrated assembler.
  if (process.platform === 'darwin' && process.arch === 'x64') {
    if (run('nasm', ['-v'], { stdio: 'pipe' }).status !== 0) {
      fail('缺少 nasm（x64 汇编需要）。请先运行: brew install nasm')
    }
  }

  const jobs = String(Math.max(2, os.cpus().length))
  const buildEnv = { ...process.env, MACOSX_DEPLOYMENT_TARGET: '11.0' }

  const x264Prefix = path.join(prefix, 'x264')
  log(`编译 x264 ${X264_COMMIT.slice(0, 8)} -> ${x264Prefix}`)
  let result = run(
    path.join(x264Src, 'configure'),
    [`--prefix=${x264Prefix}`, '--enable-static', '--disable-cli', '--disable-opencl'],
    { cwd: x264Src, stdio: 'inherit', env: buildEnv },
  )
  if (result.status !== 0) fail('x264 configure 失败')
  result = run('make', [`-j${jobs}`], { cwd: x264Src, stdio: 'inherit', env: buildEnv })
  if (result.status !== 0) fail('x264 编译失败')
  result = run('make', ['install'], { cwd: x264Src, stdio: 'inherit', env: buildEnv })
  if (result.status !== 0) fail('x264 install 失败')

  log(`编译 ffmpeg ${FFMPEG_VERSION} -> ${binDir}`)
  result = run(
    path.join(ffmpegSrc, 'configure'),
    [
      `--prefix=${path.join(prefix, 'ffmpeg')}`,
      '--enable-gpl',
      '--enable-libx264',
      // Explicitly opt into zlib (png decode); everything else in-tree is
      // built by default while --disable-autodetect keeps random host libs
      // out of the relocatable binary.
      '--enable-zlib',
      '--disable-autodetect',
      '--disable-shared',
      '--enable-static',
      '--disable-ffplay',
      '--disable-ffprobe',
      '--disable-doc',
      `--extra-cflags=-I${path.join(x264Prefix, 'include')}`,
      `--extra-ldflags=-L${path.join(x264Prefix, 'lib')}`,
    ],
    {
      cwd: ffmpegSrc,
      stdio: 'inherit',
      env: { ...buildEnv, PKG_CONFIG_PATH: path.join(x264Prefix, 'lib', 'pkgconfig') },
    },
  )
  if (result.status !== 0) fail('ffmpeg configure 失败')
  result = run('make', [`-j${jobs}`, 'ffmpeg'], { cwd: ffmpegSrc, stdio: 'inherit', env: buildEnv })
  if (result.status !== 0) fail('ffmpeg 编译失败')
  return path.join(ffmpegSrc, 'ffmpeg')
}

async function extractTarball(archive, destDir, topLevelExpect) {
  const result = run('tar', ['-xf', archive, '-C', destDir])
  if (result.status !== 0) fail(`解压失败: ${archive}`)
  if (!existsSync(path.join(destDir, topLevelExpect))) {
    fail(`压缩包布局异常: 未找到 ${topLevelExpect}`)
  }
}

async function buildFfmpeg(targetKey) {
  const targetName = isWindows(targetKey) ? 'ffmpeg.exe' : 'ffmpeg'
  const target = path.join(binDir, targetName)
  const markerFile = path.join(binDir, '.ffmpeg-build.json')
  const marker = { ...ffmpegMarker(targetKey), target: targetKey }

  if (existsSync(markerFile) && existsSync(target)) {
    try {
      const current = JSON.parse(readFileSync(markerFile, 'utf8'))
      const unchanged = JSON.stringify(current) === JSON.stringify(marker)
      if (unchanged) {
        try {
          verifyFfmpegBinary(target, targetKey)
          log(`ffmpeg 已是最新，跳过 (${FFMPEG_VERSION})`)
          return
        } catch {
          log('现有 ffmpeg 自检失败，重新构建')
        }
      }
    } catch {
      log('ffmpeg 构建标记损坏，重新构建')
    }
  }

  mkdirSync(binDir, { recursive: true })

  if (isWindows(targetKey)) {
    const cached = path.join(cacheDir, `ffmpeg-${FFMPEG_VERSION}-win64.zip`)
    await fetchVerified(FFMPEG_WIN_URL, cached, FFMPEG_WIN_SHA256)
    const extractDir = path.join(buildDir, 'ffmpeg-extract')
    rmSync(extractDir, { recursive: true, force: true })
    mkdirSync(extractDir, { recursive: true })
    await extractTarball(cached, extractDir, FFMPEG_WIN_ENTRY)
    copyFileSync(path.join(extractDir, FFMPEG_WIN_ENTRY), target)
    rmSync(extractDir, { recursive: true, force: true })
  } else {
    const ffmpegArchive = path.join(cacheDir, `ffmpeg-${FFMPEG_VERSION}.tar.xz`)
    const x264Archive = path.join(cacheDir, `x264-${X264_COMMIT}.tar.gz`)
    await fetchVerified(FFMPEG_SOURCE_URL, ffmpegArchive, FFMPEG_SOURCE_SHA256)
    await fetchVerified(X264_URL, x264Archive, X264_SHA256)

    // One scratch dir per build; both tarballs extract their own top-level
    // directory into it.
    const buildRoot = path.join(buildDir, 'ffmpeg-build')
    rmSync(buildRoot, { recursive: true, force: true })
    mkdirSync(buildRoot, { recursive: true })
    await extractTarball(ffmpegArchive, buildRoot, `ffmpeg-${FFMPEG_VERSION}`)
    await extractTarball(x264Archive, buildRoot, `x264-${X264_COMMIT}`)

    const built = await buildFfmpegFromSource(buildRoot)
    copyFileSync(built, target)
    rmSync(buildRoot, { recursive: true, force: true })
  }

  chmodSync(target, 0o755)
  verifyFfmpegBinary(target, targetKey)
  writeFileSync(markerFile, `${JSON.stringify(marker, null, 2)}\n`)
}

// ---------------------------------------------------------------------------
// Notices
// ---------------------------------------------------------------------------

function writeNotices() {
  const content = `# THIRD-PARTY NOTICES

This desktop bundle redistributes the following third-party artifacts.
Versions and sources are pinned in \`client/scripts/build-python-runtime.mjs\`.

## CPython (python-build-standalone)

- Artifact: cpython-${PBS_PYTHON_VERSION}+${PBS_RELEASE_TAG} (${PBS_VARIANT}), per-platform tarballs
- Source: https://github.com/astral-sh/python-build-standalone/releases/tag/${PBS_RELEASE_TAG}
- License: Python Software Foundation License Version 2
  (https://docs.python.org/3/license.html). Build tooling for
  python-build-standalone is MIT licensed.

## FFmpeg

- Artifact: FFmpeg ${FFMPEG_VERSION}
  - macOS (arm64/x64): compiled from the official release tarball
    (https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz) together
    with x264 ${X264_COMMIT} from https://code.videolan.org/videolan/x264
    (stable branch, pinned commit). The resulting binary links macOS system
    frameworks only.
  - Windows (x64): gyan.dev "essentials" build from
    https://github.com/GyanD/codexffmpeg/releases/tag/${FFMPEG_VERSION}
- License: GPL-2.0-or-later (the builds include libx264, which the
  application's video renderer requires). See https://ffmpeg.org/legal.html
- Source tarballs for the macOS builds are available from ffmpeg.org and
  https://code.videolan.org/videolan/x264 respectively.

## Python packages

Production dependencies are installed from \`server/requirements.lock\`
(hash-pinned). Each package's license is stated in its metadata
(\`<dist-info>/LICENSE*\` inside \`resources/python/lib/python3.11/site-packages\`
or \`resources/python/Lib/site-packages\` on Windows).
`
  writeFileSync(path.join(buildDir, 'THIRD-PARTY-NOTICES.md'), content)
  log('已生成 THIRD-PARTY-NOTICES.md')
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const args = parseArgs(process.argv.slice(2))
const targetKey = resolveTargetKey(args)

if (args.clean) {
  log(`--clean: 删除 ${buildDir}`)
  removeRecursiveIfExist(buildDir)
}

mkdirSync(cacheDir, { recursive: true })
await buildPythonRuntime(targetKey)
await buildFfmpeg(targetKey)
writeNotices()

log(`完成: ${pythonDir}`)
log(`完成: ${path.join(binDir, isWindows(targetKey) ? 'ffmpeg.exe' : 'ffmpeg')}`)
