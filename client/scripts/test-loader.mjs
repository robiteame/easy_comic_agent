// 组件测试 loader 的注册入口（通过 --import 挂载，钩子实现见 test-loader-hooks.mjs）。
import { register } from 'node:module'

register(new URL('./test-loader-hooks.mjs', import.meta.url))
