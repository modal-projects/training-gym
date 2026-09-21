import assert from 'node:assert/strict';
import { test } from 'node:test';
import { nextPaneScrollTop, sidebarScrollKey } from './sidebar-scroll.ts';

test('sidebar scroll keys are section-scoped', () => {
  assert.equal(sidebarScrollKey('SDK'), 'tg-sidebar-scroll:SDK');
  assert.notEqual(sidebarScrollKey('SDK'), sidebarScrollKey('CLI'));
});

test('visible sidebar items keep the restored scrollTop', () => {
  assert.equal(nextPaneScrollTop(400, 200, 450, 24), 400);
});

test('items above the pane snap to their top', () => {
  assert.equal(nextPaneScrollTop(400, 200, 100, 24), 100);
});

test('items below the pane snap just into view', () => {
  assert.equal(nextPaneScrollTop(0, 200, 800, 24), 624);
});
