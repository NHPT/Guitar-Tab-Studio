import assert from 'node:assert/strict'
import test from 'node:test'
import {
  developmentRendererUrl,
  isDesktopNavigationAllowed,
  isUserRendererPath,
} from '../src/desktop-navigation.js'

test('accepts only user-facing renderer paths', () => {
  assert.equal(isUserRendererPath('/'), true)
  assert.equal(isUserRendererPath('/workspace'), true)
  assert.equal(isUserRendererPath('/admin'), false)
  assert.equal(isUserRendererPath('/admin/reviews/beat'), false)
  assert.equal(isUserRendererPath('/review'), false)
  assert.equal(isUserRendererPath('/community/tasks'), false)
  assert.equal(isUserRendererPath('/%61dmin'), false)
})

test('limits renderer overrides to loopback user pages', () => {
  assert.equal(
    developmentRendererUrl('http://127.0.0.1:5173/')?.origin,
    'http://127.0.0.1:5173',
  )
  assert.throws(
    () => developmentRendererUrl('http://127.0.0.1:5173/admin'),
    /user-facing loopback/u,
  )
  assert.throws(
    () => developmentRendererUrl('https://example.com/'),
    /user-facing loopback/u,
  )
})

test('allows navigation only within the exact desktop entry', () => {
  const fileEntry = 'file:///opt/Guitar%20Tab%20Studio/web/desktop.html'
  assert.equal(isDesktopNavigationAllowed(fileEntry, fileEntry), true)
  assert.equal(
    isDesktopNavigationAllowed(
      `${fileEntry}#/admin`,
      fileEntry,
    ),
    false,
  )
  assert.equal(
    isDesktopNavigationAllowed(
      'file:///opt/Guitar%20Tab%20Studio/web/admin.html',
      fileEntry,
    ),
    false,
  )

  const developmentEntry = 'http://127.0.0.1:5173/'
  assert.equal(
    isDesktopNavigationAllowed(
      'http://127.0.0.1:5173/?project=demo',
      developmentEntry,
    ),
    true,
  )
  assert.equal(
    isDesktopNavigationAllowed(
      'http://127.0.0.1:5173/admin',
      developmentEntry,
    ),
    false,
  )
})
