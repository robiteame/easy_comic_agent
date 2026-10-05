// zustand 的测试期 React 绑定：store 本体复用真实的 zustand/vanilla
// （setState/subscribe/getState 语义不变），仅替换 hook 绑定。
//
// 原因：zustand v4 的 useStore 在 SSR（getServerSnapshot）读取内部 api 的
// getInitialState，即模块加载时的初始状态，测试注入的 setState 状态渲染
// 不出来，且内部 api 对象无法从外部打补丁。这里的绑定直接以 getState
// 作为 server snapshot，使 renderToString 反映测试注入的实时状态。
import * as React from 'react'
import { createStore as vanillaCreateStore } from 'zustand/vanilla'

const identity = (arg) => arg

function useStore(api, selector = identity) {
  return React.useSyncExternalStore(
    api.subscribe,
    () => selector(api.getState()),
    () => selector(api.getState()),
  )
}

function createImpl(createState) {
  const api = vanillaCreateStore(createState)
  const useBoundStore = (selector) => useStore(api, selector)
  Object.assign(useBoundStore, api)
  return useBoundStore
}

export function create(createState) {
  return createState ? createImpl(createState) : createImpl
}

export { useStore }
export * from 'zustand/vanilla'
