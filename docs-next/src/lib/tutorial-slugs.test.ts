import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { discoverTutorialEntries } from './tutorial-slugs.ts';

test('rejects flat and nested tutorials with the same slug', async (context) => {
  const directory = await mkdtemp(path.join(tmpdir(), 'tutorial-slugs-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(path.join(directory, 'example.py'), '');
  await mkdir(path.join(directory, 'example'));
  await writeFile(path.join(directory, 'example', 'main.py'), '');

  await assert.rejects(
    discoverTutorialEntries(directory),
    /Tutorial slug "example" is defined by both/,
  );
});
