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

for (const name of ['has.dot', 'has space']) {
  test(`rejects folder name ${JSON.stringify(name)}`, async (context) => {
    const directory = await mkdtemp(path.join(tmpdir(), 'tutorial-slugs-'));
    context.after(() => rm(directory, { recursive: true, force: true }));
    await mkdir(path.join(directory, name));
    await writeFile(path.join(directory, name, 'main.py'), '');

    await assert.rejects(
      discoverTutorialEntries(directory),
      /not a valid Python module name/,
    );
  });
}

test('nested sourcePath is a directory and runTarget is a module', async (context) => {
  const directory = await mkdtemp(path.join(tmpdir(), 'tutorial-slugs-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(path.join(directory, 'flat.py'), '');
  await writeFile(path.join(directory, 'main.py'), '');
  await mkdir(path.join(directory, 'nested'));
  await writeFile(path.join(directory, 'nested', 'main.py'), '');

  const entries = Object.fromEntries(
    (await discoverTutorialEntries(directory)).map((entry) => [entry.slug, entry]),
  );

  assert.equal(entries.flat?.runTarget, 'tutorials/flat.py');
  assert.equal(entries.flat?.sourcePath, 'tutorials/flat.py');
  assert.equal(entries.main?.runTarget, 'tutorials/main.py');
  assert.equal(entries.main?.sourcePath, 'tutorials/main.py');
  assert.equal(entries.nested?.runTarget, '-m tutorials.nested.main');
  assert.equal(entries.nested?.sourcePath, 'tutorials/nested');
});
