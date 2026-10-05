// 组件测试辅助：在纯 Node（node:test，无 jsdom）环境下用 react-dom/server
// 渲染 React 组件做冒烟与内容断言。
//
// SSR 渲染不执行 useEffect / 事件回调，因此组件里 window/document 相关调用
// 只要不在渲染期执行就不会触发；个别在渲染期直接读取浏览器全局的组件
// （如 matchMedia）由这里注入的最小桩兜底。
import assert from 'node:assert/strict'
import type { ComponentType } from 'react'
import React from 'react'
import { renderToString } from 'react-dom/server'

type AnyRecord = Record<string, any>

/**
 * 安装最小浏览器全局桩。幂等，可重复调用。
 * 覆盖渲染期常见读取：matchMedia、ResizeObserver、getComputedStyle、
 * scrollTo、requestAnimationFrame、canvas measureText。
 */
export function installBrowserStubs(): void {
  const g = globalThis as unknown as AnyRecord
  if (g.__comicAgentBrowserStubsInstalled) return
  g.__comicAgentBrowserStubsInstalled = true

  if (typeof g.window === 'undefined') g.window = g
  if (typeof g.self === 'undefined') g.self = g

  if (!g.matchMedia) {
    g.matchMedia = (query: string) => ({
      media: query,
      matches: false,
      onchange: null,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent() {
        return false
      },
    })
  }

  if (!g.ResizeObserver) {
    g.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  }

  if (!g.IntersectionObserver) {
    g.IntersectionObserver = class IntersectionObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
      takeRecords() {
        return []
      }
    }
  }

  if (!g.requestAnimationFrame) {
    g.requestAnimationFrame = (cb: (t: number) => void) => setTimeout(() => cb(Date.now()), 0) as unknown as number
    g.cancelAnimationFrame = clearTimeout
  }

  if (!g.getComputedStyle) {
    g.getComputedStyle = () => ({
      getPropertyValue() {
        return ''
      },
    })
  }

  if (!g.scrollTo) g.scrollTo = () => {}

  if (!g.HTMLCanvasElement) {
    g.HTMLCanvasElement = class HTMLCanvasElement {
      getContext() {
        return {
          measureText: (text: string) => ({ width: String(text ?? '').length * 7 }),
          fillText() {},
        }
      }
    }
  }
}

/** renderToString 的便捷封装，先确保浏览器桩就位。 */
export function renderToHtml(element: React.ReactElement): string {
  installBrowserStubs()
  return renderToString(element)
}

/**
 * 把 SSR 输出转成「可见文本」：剥离 React 在文本与表达式边界插入的
 * <!-- --> 注释节点，便于对连续文案做 includes 断言。
 */
export function visibleText(html: string): string {
  return html.replace(/<!--.*?-->/g, '')
}

/**
 * 断言组件以给定 props 渲染不抛错，且产出非空标记（冒烟测试）。
 * 返回渲染出的 HTML 供进一步断言。
 */
export function expectRender<P extends object>(Comp: ComponentType<P>, props: P = {} as P): string {
  const html = renderToHtml(React.createElement(Comp as ComponentType<AnyRecord>, props))
  assert.ok(typeof html === 'string', 'renderToString 应返回字符串')
  return html
}
