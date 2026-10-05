# AI漫剧Agent · 前端技术文档

## 项目概述

AI漫剧Agent 前端是一个基于 **Electron + React + TypeScript** 的桌面客户端，采用 Mac 原生极简设计风格，提供五栏工作台布局，支持剧本输入、分镜预览、镜头编辑、参数配置和导出渲染等完整工作流。

**核心定位**：五栏工作台 → 剧本驱动 → 实时预览 → 一键出片

**代码规范**：TS/TSX/JSON 由 Biome 统一 lint + format（配置在仓库根 `biome.json`：单引号、JSX 属性双引号、无分号、2 空格、行宽 120；CSS 暂不格式化）。提交前 lefthook 会自动修复暂存文件；CI 中 `biome check client package.json` 为门禁。改动客户端代码后在仓库根跑 `pnpm lint`（自动修复用 `pnpm lint:fix`）。

---

## 技术栈

| 组件 | 选型 | 版本 | 用途 |
|------|------|------|------|
| 桌面框架 | Electron | 32.x | 桌面客户端壳体 |
| 前端框架 | React | 18.3 | UI 渲染 |
| 构建工具 | Vite | 5.4 | 开发服务器 + 构建打包 |
| UI 组件库 | Ant Design | 5.20 | 表单、按钮、选择器等基础组件 |
| 状态管理 | Zustand | 4.5 | 轻量全局状态 (项目/镜头) |
| 拖拽排序 | @dnd-kit | 6.x / 8.x | 镜头拖拽排序 |
| HTTP 客户端 | Axios | 1.7 | REST API 调用 |
| 音频波形 | wavesurfer.js | 7.8 | 配音波形可视化 |
| 类型检查 | TypeScript | 5.6 | 静态类型 |
| 语言 | 中文 (zhCN) | - | Ant Design 国际化 |

---

## 目录结构

```
client/
├── index.html                     # 入口 HTML
├── package.json                   # 依赖与脚本
├── tsconfig.json                  # TypeScript 配置
├── tsconfig.node.json             # Vite Node 配置
├── vite.config.ts                 # Vite + Electron 插件配置
├── .npmrc                         # Electron 国内镜像源
│
├── src/
│   ├── main/                      # Electron 主进程
│   │   ├── main.ts                # BrowserWindow 创建、IPC 处理
│   │   └── preload.ts             # contextBridge 暴露安全 API
│   │
│   └── renderer/                  # React 渲染进程
│       ├── main.tsx               # React 入口 + Ant Design 主题配置
│       ├── App.tsx                # 五栏布局根组件
│       │
│       ├── components/            # 布局组件
│       │   ├── TopBar.tsx         # 顶栏：标题 + 操作按钮 + 设置
│       │   ├── LeftSidebar.tsx    # 左栏：项目列表 + Agent 执行流程
│       │   ├── MainWorkspace.tsx  # 主区：剧本输入 + 画面预览 + 镜头缩略图
│       │   ├── RightSidebar.tsx   # 右栏：风格设置 + 镜头控制 + 运行日志
│       │   └── BottomBar.tsx      # 底栏：任务进度 + 时长 + GPU 状态
│       │
│       ├── stores/                # Zustand 状态管理
│       │   ├── projectStore.ts    # 项目状态 (ID/标题/风格/分辨率等)
│       │   └── shotStore.ts       # 镜头状态 (列表/选中/生成进度)
│       │
│       ├── services/              # 后端通信层
│       │   └── api.ts             # Axios 封装 + WebSocket 工厂
│       │
│       └── styles/
│           └── global.css         # Mac 原生极简设计系统
│
└── dist-electron/                 # Electron 主进程构建产物 (gitignore)
```

---

## 布局架构

五栏工作台，使用纯 Flexbox 布局，无路由：

```
┌──────────────────────────────────────────────────┐
│                      TopBar (44px)                │
├──────────┬───────────────────────┬────────────────┤
│          │                       │                │
│   Left   │    MainWorkspace      │     Right      │
│  Sidebar │  ┌─────────────────┐  │    Sidebar     │
│  (240px) │  │   剧本输入区    │  │    (300px)     │
│          │  │    (32%)        │  │                │
│  项目列表│  ├─────────────────┤  │   风格设置     │
│          │  │                 │  │                │
│  Agent   │  │   画面预览区    │  │   运行模式     │
│  流程    │  │   (flex: 1)     │  │                │
│          │  │  [缩略图条]     │  │   镜头控制     │
│          │  └─────────────────┘  │                │
│          │                       │   运行日志     │
├──────────┴───────────────────────┴────────────────┤
│                    BottomBar (32px)                │
└──────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 宽高 | 职责 |
|------|------|------|
| TopBar | 44px, 全宽 | 项目标题、新建/导入/生成/导出按钮、设置/帮助入口 |
| LeftSidebar | 240px | 项目列表切换、Agent 5 步执行流程状态指示灯 |
| MainWorkspace | flex:1 | 上部剧本编辑区 + 下部画面预览区 + 底部镜头缩略图条 |
| RightSidebar | 300px | 画风/分辨率选择、Agent 运行模式、当前镜头详情、运行日志 |
| BottomBar | 32px, 全宽 | 任务进度条、预估剩余时间、总时长、GPU 占用 |

---

## 状态管理

### projectStore — 项目状态

```typescript
interface ProjectState {
  projectId: string | null
  title: string          // 项目名称
  genre: string          // 类型 (甜宠/悬疑/古风/都市)
  style: string          // 画风 (anime/chinese/chibi/realistic)
  status: string         // draft/generating/completed/error
  outputFormat: string   // 9:16 / 16:9 / 1:1
  resolution: string     // 720p / 1080p / 2k / 4k
  platform: string       // douyin / kuaishou / bilibili / custom
}
```

**默认值**: style=`anime`, outputFormat=`9:16`, resolution=`1080p`, platform=`douyin`

### shotStore — 镜头状态

```typescript
interface Shot {
  id: string
  project_id: string
  sequence: number
  shot_type: string        // wide / medium / close-up / extreme_close
  scene_description: string
  character_action: string
  dialogue: string
  camera_angle: string
  duration: number         // 秒
  emotion: string
  transition: string
  image_path: string       // 后端图片路径
  audio_path: string       // 后端音频路径
  status: string           // pending / generating / done / failed / needs_review
  version: number
  characters_in_scene: string[]
}
```

**关键操作**:
- `setShots` — 批量设置 (WebSocket complete 事件)
- `updateShot` — 单镜头更新
- `reorderShots` — 拖拽排序 (dnd-kit 回调)
- `setProgress` — 更新进度条 + 当前步骤

---

## 后端通信

### REST API 封装

`services/api.ts` 提供 5 组 API 对象：

| 对象 | 方法 | 对应后端路由 |
|------|------|-------------|
| `projectApi` | create / get / list / update | `/api/project` |
| `scriptApi` | parse / upload | `/api/script` |
| `shotApi` | list / update / regenerate / batchRegenerate | `/api/shot` |
| `renderApi` | start / status | `/api/render` |
| `chatApi` | send | `/api/chat` |

**后端地址**: `http://localhost:8000` (写死在 api.ts 顶部)

### WebSocket

```typescript
createWebSocket(projectId, onMessage) → WebSocket
```

- **地址**: `ws://localhost:8000/ws/{project_id}`
- **心跳**: 每 30 秒发送 `ping`
- **监听事件**: `progress` / `complete` / `shot_update` / `render_complete` / `error`
- **complete 事件携带 `title`**: 剧本解析完成后后端已根据剧本自动命名项目，
  MainWorkspace 收到后同步标题输入框并派发 `PROJECTS_REFRESHED_EVENT`
  （`constants/events.ts`）刷新侧边栏项目列表。

### 任务启动闸门（与后端 409 约定）

提交任务的入口在失败时先解析两类结构化拦截（HTTP 409 + `detail` 对象），命中则弹
通知并短路，不再走通用错误提示：

- `budget_exceeded`（硬预算不足）→ `notifyBudgetBlocked`；
- `provider_not_configured`（模型端点未配置 API Key）→ `notifyProviderBlocked`，
  通知附「去系统设置」按钮，跳转模型服务配置页。
  解析函数在 `services/costModel.ts`（`providerBlockedFromError`），展示在
  `components/TaskEstimateModal.tsx`。

### 完整生成流程

```
用户输入剧本 → 点击"生成分镜"
  │
  ├─ 1. projectApi.create()      → 创建项目，获得 projectId
  ├─ 2. createWebSocket()        → 建立 WS 连接，监听进度
  └─ 3. scriptApi.parse()        → 提交剧本，触发后端 Agent 流水线
       │
       ├─ WS: progress 事件      → 更新进度条 + 当前步骤
       ├─ WS: complete 事件      → 写入 shots，关闭 WS
       └─ WS: error 事件         → 显示错误，关闭 WS
```

---

## Electron 主进程

### main.ts

- **窗口**: 1400x900, 最小 1200x800
- **开发模式**: 加载 `http://localhost:5173`，自动打开 DevTools
- **生产模式**: 加载 `dist/index.html`
- **IPC 通道**:
  - `select-file` — 打开文件选择对话框 (txt/docx/*)
  - `select-directory` — 打开目录选择对话框

### preload.ts

通过 `contextBridge` 安全暴露：

```typescript
window.electronAPI = {
  selectFile: () => string | null
  selectDirectory: () => string | null
}
```

---

## 设计系统

### 色彩体系 (Apple 风格)

| 变量 | 值 | 用途 |
|------|------|------|
| `--accent` | `#0071e3` | 主色 (按钮/选中/进度条) |
| `--bg` | `#f5f5f7` | 页面背景 |
| `--bg-white` | `#ffffff` | 卡片/面板背景 |
| `--bg-canvas` | `#f9f9fb` | 画布/预览区背景 |
| `--text` | `#1d1d1f` | 主文字 |
| `--text-secondary` | `#6e6e73` | 次级文字 |
| `--text-tertiary` | `#aeaeb2` | 占位符/禁用 |
| `--border` | `#e0e0e0` | 边框 |
| `--green` | `#34c759` | 成功/完成 |
| `--amber` | `#ff9500` | 运行中/警告 |
| `--red` | `#ff3b30` | 错误/失败 |

### 字体

```css
font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
             "Helvetica Neue", "Noto Sans SC", sans-serif;
```

### Ant Design 主题覆盖

全局覆盖了以下组件样式以适配 Mac 原生风格：
- **Button**: 30px 高度, 6px 圆角, 无 box-shadow
- **Input/Select**: 30px 高度, 聚焦时无阴影
- **Menu**: 透明背景, 32px 项高, 选中态蓝色背景
- **Tag**: 无边框, 4px 圆角
- **Slider**: 蓝色轨道, 白色把手
- **Progress**: 蓝色进度条
- **Upload**: 虚线边框拖拽区

### 动画

```css
@keyframes fadeIn    { opacity: 0 → 1, translateY 4px → 0 }
@keyframes shimmer   { translateX -100% → 300% }  /* 加载进度条 */
```

---

## 开发脚本

| 命令 | 说明 |
|------|------|
| `pnpm dev` | 启动 Vite 开发服务器 (仅前端) |
| `pnpm electron:dev` | 同时启动 Vite + Electron (完整开发) |
| `pnpm build` | TypeScript 检查 + Vite 构建 + Electron 打包 |
| `pnpm electron:build` | Vite 构建 + Electron 打包 |

---

## 开发环境搭建

```bash
cd client

# 1. 安装依赖 (需要 pnpm)
pnpm install

# 2. 启动开发
pnpm electron:dev
```

### 前置条件

- **Node.js** >= 18
- **pnpm** (推荐) 或 npm
- **后端服务** 运行在 `localhost:8000` (否则 API 和 WebSocket 不可用)

### Electron 安装问题

如果 Electron 二进制下载失败，`.npmrc` 已配置国内镜像：

```
electron_mirror=https://npmmirror.com/mirrors/electron/
electron_builder_binaries_mirror=https://npmmirror.com/mirrors/electron-builder-binaries/
```

若仍失败，手动设置：

```bash
npx electron-builder install-app-deps
```

---

## 待实现组件

以下组件已在 `App.tsx` 中引用并已实现，但仍有功能可扩展：

| 组件 | 当前状态 | 可扩展方向 |
|------|---------|-----------|
| TopBar | 基础按钮框架 | 导入剧本文件、设置弹窗、帮助文档 |
| LeftSidebar | 项目列表 + 步骤指示 | 项目切换加载、步骤点击跳转 |
| MainWorkspace | 剧本输入 + 图片预览 | dnd-kit 拖拽排序缩略图、波形播放 |
| RightSidebar | 风格选择 + 信息展示 | 运镜参数联动后端、镜头编辑表单 |
| BottomBar | 进度条 + 时长 | 真实 GPU 监控、渲染队列 |

---

## 路径别名

`@` 映射到 `src/renderer/`，可在任何渲染进程文件中使用：

```typescript
import { useProjectStore } from '@/stores/projectStore'
```
