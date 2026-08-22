import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { docsLoader } from '@astrojs/starlight/loaders';
import type { Loader, LoaderContext } from 'astro/loaders';
import { TUTORIAL_ENTRY_PREFIX } from './tutorial-page';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const tutorialsDirectory = path.join(repoRoot, 'tutorials');
const pyprojectPath = path.join(repoRoot, 'pyproject.toml');
const frontmatterFieldPattern = /^# ([a-z_]+):\s*(.*)$/;
const dependencyPattern = /^[A-Za-z0-9_.-]+$/;

export interface Tutorial {
  path: string;
  slug: string;
  order: number;
  title: string;
  body: string;
  runCommand: string;
}

function markdownFromComments(lines: string[]): string {
  return lines
    .map((line) => (line === '#' ? '' : line.slice(2)))
    .join('\n')
    .trim();
}

function renderBody(source: string, tutorialPath: string): { title: string; body: string } {
  const lines = source.split('\n');
  const blocks: Array<{ kind: 'markdown' | 'code'; content: string }> = [];
  let index = 0;

  while (index < lines.length) {
    const isMarkdown = lines[index] === '#' || lines[index].startsWith('# ');
    const start = index;
    while (
      index < lines.length &&
      (lines[index] === '#' || lines[index].startsWith('# ')) === isMarkdown
    ) {
      index += 1;
    }
    const blockLines = lines.slice(start, index);
    const content = isMarkdown
      ? markdownFromComments(blockLines)
      : blockLines.join('\n').trim();
    if (content) {
      blocks.push({ kind: isMarkdown ? 'markdown' : 'code', content });
    }
  }

  const titleLine = blocks
    .filter((block) => block.kind === 'markdown')
    .flatMap((block) => block.content.split('\n'))
    .find((line) => line.startsWith('# '));
  if (!titleLine) {
    throw new Error(`${tutorialPath} is missing an H1 heading`);
  }
  const title = titleLine.slice(2).trim();
  let removedTitle = false;
  const body = blocks
    .map((block) => {
      if (block.kind === 'code') {
        return `\`\`\`python\n${block.content}\n\`\`\``;
      }
      if (!removedTitle && block.content.split('\n').includes(titleLine)) {
        removedTitle = true;
        return block.content
          .split('\n')
          .filter((line) => line !== titleLine)
          .join('\n')
          .trim();
      }
      return block.content;
    })
    .filter(Boolean)
    .join('\n\n');

  return { title, body };
}

function parseFrontmatter(
  source: string,
  tutorialPath: string,
): { order: number; deps: string[]; content: string } {
  const lines = source.split('\n');
  if (lines[0] !== '# ---') {
    throw new Error(`${tutorialPath} must start with tutorial frontmatter`);
  }
  const frontmatterEnd = lines.indexOf('# ---', 1);
  if (frontmatterEnd === -1) {
    throw new Error(`${tutorialPath} has unterminated tutorial frontmatter`);
  }

  const fields = new Map<string, string>();
  for (const line of lines.slice(1, frontmatterEnd)) {
    const match = line.match(frontmatterFieldPattern);
    if (!match) {
      throw new Error(`${tutorialPath} has invalid frontmatter line: ${line}`);
    }
    const [, name, value] = match;
    if (name !== 'order' && name !== 'deps') {
      throw new Error(`${tutorialPath} has unsupported frontmatter field: ${name}`);
    }
    if (fields.has(name)) {
      throw new Error(`${tutorialPath} has duplicate frontmatter field: ${name}`);
    }
    fields.set(name, value);
  }

  const orderText = fields.get('order');
  if (!orderText || !/^\d+$/.test(orderText)) {
    throw new Error(`${tutorialPath} frontmatter requires a non-negative integer order`);
  }
  const deps = (fields.get('deps') ?? '')
    .split(',')
    .map((dependency) => dependency.trim())
    .filter(Boolean);
  if (new Set(deps).size !== deps.length) {
    throw new Error(`${tutorialPath} frontmatter deps must be unique`);
  }
  const invalidDeps = deps.filter((dependency) => !dependencyPattern.test(dependency));
  if (invalidDeps.length > 0) {
    throw new Error(`${tutorialPath} has invalid frontmatter deps: ${invalidDeps.join(', ')}`);
  }

  return {
    order: Number(orderText),
    deps,
    content: lines.slice(frontmatterEnd + 1).join('\n'),
  };
}

function formatRunCommand(slug: string, extras: string[]): string {
  if (extras.length === 0) {
    return `uv run tutorials/${slug}.py`;
  }
  return `uv run ${extras.map((pkg) => `--with ${pkg}`).join(' ')} tutorials/${slug}.py`;
}

async function readTutorial(fileName: string): Promise<Tutorial> {
  const tutorialPath = path.join(tutorialsDirectory, fileName);
  const source = await readFile(tutorialPath, 'utf8');
  const { order, deps, content } = parseFrontmatter(source, tutorialPath);
  const slug = path.basename(fileName, '.py');
  const { title, body } = renderBody(content, tutorialPath);
  return {
    path: tutorialPath,
    slug,
    order,
    title,
    body,
    runCommand: formatRunCommand(slug, deps),
  };
}

export async function loadTutorials(): Promise<Tutorial[]> {
  const fileNames = (await readdir(tutorialsDirectory))
    .filter((fileName) => fileName.endsWith('.py'))
    .sort();
  const tutorials = await Promise.all(fileNames.map((fileName) => readTutorial(fileName)));
  tutorials.sort((left, right) => left.order - right.order || left.slug.localeCompare(right.slug));
  const expectedOrders = tutorials.map((_, index) => index);
  const actualOrders = tutorials.map((tutorial) => tutorial.order);
  if (actualOrders.some((order, index) => order !== expectedOrders[index])) {
    throw new Error(`Tutorial orders must be contiguous from 0: ${actualOrders.join(', ')}`);
  }
  return tutorials;
}

async function storeEntry(
  context: LoaderContext,
  id: string,
  data: Record<string, unknown>,
  body: string,
  filePath?: string,
): Promise<void> {
  const parsedData = await context.parseData({ id, data, filePath });
  context.store.set({
    id,
    data: parsedData,
    body,
    filePath,
    digest: context.generateDigest(body),
    rendered: await context.renderMarkdown(body),
  });
}

export function tutorialDocsLoader(): Loader {
  const starlightDocsLoader = docsLoader();
  let tutorialEntryIds = new Set<string>();

  async function syncTutorialEntries(context: LoaderContext): Promise<void> {
    const tutorials = await loadTutorials();
    const nextEntryIds = new Set<string>();
    for (const tutorial of tutorials) {
      const id = `${TUTORIAL_ENTRY_PREFIX}${tutorial.slug}`;
      nextEntryIds.add(id);
      await storeEntry(
        context,
        id,
        {
          title: tutorial.title,
          sidebar: { order: tutorial.order },
          next: tutorial.order === tutorials.length - 1 ? false : undefined,
          runCommand: tutorial.runCommand,
        },
        tutorial.body,
      );
    }
    for (const id of tutorialEntryIds) {
      if (!nextEntryIds.has(id)) {
        context.store.delete(id);
      }
    }
    tutorialEntryIds = nextEntryIds;
  }

  return {
    name: 'training-gym-tutorial-docs-loader',
    async load(context) {
      await starlightDocsLoader.load(context);
      await syncTutorialEntries(context);

      const { watcher } = context;
      if (!watcher) {
        return;
      }
      watcher.add([tutorialsDirectory, pyprojectPath]);

      const isTutorialSource = (changedPath: string) => {
        const resolvedPath = path.resolve(changedPath);
        return (
          path.dirname(resolvedPath) === tutorialsDirectory &&
          path.extname(resolvedPath) === '.py'
        );
      };
      let pendingReload = Promise.resolve();
      const reloadTutorials = (changedPath: string) => {
        const resolvedPath = path.resolve(changedPath);
        if (resolvedPath !== pyprojectPath && !isTutorialSource(resolvedPath)) {
          return;
        }
        pendingReload = pendingReload
          .catch(() => undefined)
          .then(async () => {
            await syncTutorialEntries(context);
            context.logger.info(`Reloaded tutorials after ${path.basename(resolvedPath)} changed`);
          });
        return pendingReload;
      };
      watcher.on('add', reloadTutorials);
      watcher.on('change', reloadTutorials);
      watcher.on('unlink', reloadTutorials);
    },
  };
}
