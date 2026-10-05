import {
  DEFAULT_THEME,
  THEME_OPTIONS,
  applyThemeToDocument,
  getInitialTheme,
  resolveTheme,
  type AppTheme,
} from './theme.ts'

function assert(condition: unknown, message: string) {
  if (!condition) {
    throw new Error(message)
  }
}

function createStorage(initial?: string) {
  let value = initial
  return {
    getItem: (key: string) => (key === 'comic-agent-theme' ? value || null : null),
    setItem: (key: string, next: string) => {
      if (key === 'comic-agent-theme') value = next
    },
  }
}

function createDocument() {
  const attributes: Record<string, string> = {}
  return {
    documentElement: {
      setAttribute: (name: string, value: string) => {
        attributes[name] = value
      },
      getAttribute: (name: string) => attributes[name] || null,
    },
  }
}

assert(DEFAULT_THEME === 'light', 'default theme should preserve the existing light UI')
assert(
  THEME_OPTIONS.some((item) => item.value === 'black'),
  'black theme option should be available',
)
assert(resolveTheme('black') === 'black', 'black theme should resolve as black')
assert(resolveTheme('unknown') === DEFAULT_THEME, 'unknown theme should fall back to default')

const persisted = getInitialTheme(createStorage('black') as Storage)
assert(persisted === 'black', 'initial theme should read a valid persisted value')

const fallback = getInitialTheme(createStorage('invalid') as Storage)
assert(fallback === DEFAULT_THEME, 'invalid persisted value should fall back to default')

const mockDocument = createDocument()
applyThemeToDocument('black' as AppTheme, mockDocument as unknown as Document, createStorage())
assert(
  mockDocument.documentElement.getAttribute('data-theme') === 'black',
  'applying a theme should update the root data-theme attribute',
)
