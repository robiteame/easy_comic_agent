import { toOutputUrl } from './api.ts'

import { test } from 'vitest'

test('module assertions', async () => {
  function assert(condition: unknown, message: string) {
    if (!condition) throw new Error(message)
  }

  assert(
    toOutputUrl('/Users/demo/output/projects/p1/output/scene 01.png') ===
      'http://127.0.0.1:8011/output/projects/p1/output/scene%2001.png',
    'absolute output paths should preserve nested output directories and encode URLs',
  )
  assert(
    toOutputUrl('output/projects/p1/storyboard.png') === 'http://127.0.0.1:8011/output/projects/p1/storyboard.png',
    'output-rooted paths should not duplicate the output prefix',
  )
  assert(toOutputUrl('') === null, 'empty output paths should not produce a broken URL')
  assert(toOutputUrl('javascript:alert(1)') === null, 'script URLs must never become media URLs')
  assert(toOutputUrl('data:text/html,<svg/onload=alert(1)>') === null, 'data URLs must never become media URLs')
})
