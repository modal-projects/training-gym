import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  flattenDocId,
  isIdentifierTitlePath,
  REFERENCE_SURFACES,
  referenceSidebarSections,
  referenceSurfaceForPath,
  type SidebarLinkEntry,
} from './docs-sections.ts';

const generatedSidebar = JSON.parse(
  readFileSync(new URL('../generated/reference-sidebar.json', import.meta.url), 'utf8'),
) as {
  sdk: { label: string; link: string }[];
  cli: { label: string; link: string }[];
};

function linksFrom(items: { label: string; link: string }[]): SidebarLinkEntry[] {
  return items.map((item) => ({
    type: 'link',
    label: item.label,
    href: item.link,
    isCurrent: false,
  }));
}

function sidebarLabels(
  surfaceLabel: 'SDK' | 'CLI',
  path: string,
): { header: string; items: string[] } {
  const surface = REFERENCE_SURFACES.find((item) => item.label === surfaceLabel);
  assert.ok(surface);
  const entries = linksFrom(
    surfaceLabel === 'CLI' ? generatedSidebar.cli : generatedSidebar.sdk,
  );
  const sections = referenceSidebarSections(surface, entries, path);
  assert.equal(sections.length, 1);
  return {
    header: sections[0].label,
    items: sections[0].items.map((item) => item.label),
  };
}

test('CLI reference pages stay nested', () => {
  assert.equal(flattenDocId('reference/cli/run'), 'reference/cli/run');
  assert.equal(flattenDocId('reference/cli/run.md'), 'reference/cli/run');
});

test('SDK overview stays nested and grouping folders still flatten', () => {
  assert.equal(flattenDocId('reference/sdk'), 'reference/sdk');
  assert.equal(flattenDocId('reference/sdk.md'), 'reference/sdk');
  assert.equal(flattenDocId('reference/core/trainconfig'), 'reference/trainconfig');
  assert.equal(flattenDocId('reference/core/trainconfig.md'), 'reference/trainconfig');
  assert.equal(flattenDocId('reference/models/modelconfig.md'), 'reference/modelconfig');
  assert.equal(flattenDocId('reference/datasets/datasetconfig.md'), 'reference/datasetconfig');
  assert.equal(flattenDocId('reference/recipes/qwen3_4b_recipe.md'), 'reference/qwen3_4b_recipe');
  assert.equal(flattenDocId('reference/training/trainconfig.md'), 'reference/trainconfig');
  assert.equal(
    flattenDocId('reference/deployment/customdeployment.md'),
    'reference/customdeployment',
  );
});

test('reference surfaces use longest prefix and SDK lands on /reference/sdk', () => {
  assert.equal(referenceSurfaceForPath('/reference/cli')?.label, 'CLI');
  assert.equal(referenceSurfaceForPath('/reference/cli')?.href, '/reference/cli');
  assert.equal(referenceSurfaceForPath('/reference')?.label, 'SDK');
  assert.equal(referenceSurfaceForPath('/reference/sdk')?.label, 'SDK');
  assert.equal(referenceSurfaceForPath('/reference/sdk')?.href, '/reference/sdk');
  assert.equal(referenceSurfaceForPath('/reference/modelconfig')?.label, 'SDK');
  assert.equal(referenceSurfaceForPath('/reference/modelconfig')?.href, '/reference/sdk');
});

test('identifier titles are leaf CLI and SDK pages only', () => {
  assert.equal(isIdentifierTitlePath('/reference/cli'), false);
  assert.equal(isIdentifierTitlePath('/reference'), false);
  assert.equal(isIdentifierTitlePath('/reference/sdk'), false);
  assert.equal(isIdentifierTitlePath('/reference/cli/run'), true);
  assert.equal(isIdentifierTitlePath('/reference/modelconfig'), true);
  assert.equal(isIdentifierTitlePath('/guides/agent'), false);
});

test('reference sidebar renders the current surface only', () => {
  const cli = sidebarLabels('CLI', '/reference/cli');
  assert.equal(cli.header, 'CLI');
  assert.deepEqual(
    cli.items,
    generatedSidebar.cli.map((item) => item.label),
  );
  assert.equal(cli.items.includes('SDK'), false);
  assert.equal(cli.header.includes('SDK'), false);

  assert.deepEqual(sidebarLabels('CLI', '/reference/cli/run'), cli);

  const sdk = sidebarLabels('SDK', '/reference/sdk');
  assert.equal(sdk.header, 'SDK');
  assert.deepEqual(
    sdk.items,
    generatedSidebar.sdk.map((item) => item.label),
  );
  assert.deepEqual(sidebarLabels('SDK', '/reference/customdeployment'), sdk);
  assert.equal(sdk.items.includes('CLI'), false);
  assert.equal(sdk.header.includes('CLI'), false);
  assert.equal(
    sdk.items.some((label) =>
      ['Models', 'Datasets', 'Recipes', 'Training', 'Deployment'].includes(label),
    ),
    false,
  );
});

test('SDK sidebar sorts A-Z even when entries arrive in group order', () => {
  const surface = REFERENCE_SURFACES.find((item) => item.label === 'SDK');
  assert.ok(surface);
  const sections = referenceSidebarSections(
    surface,
    [
      {
        type: 'link',
        label: 'TrainConfig',
        href: '/reference/trainconfig',
        isCurrent: false,
      },
      {
        type: 'link',
        label: 'CustomDeployment',
        href: '/reference/customdeployment',
        isCurrent: false,
      },
      {
        type: 'link',
        label: 'DatasetConfig',
        href: '/reference/datasetconfig',
        isCurrent: false,
      },
    ],
    '/reference/sdk',
  );
  assert.deepEqual(
    sections[0].items.map((item) => item.label),
    ['CustomDeployment', 'DatasetConfig', 'TrainConfig'],
  );
});

test('SDK sidebar flattens groups and does not list SDK as a leaf', () => {
  const surface = REFERENCE_SURFACES.find((item) => item.label === 'SDK');
  assert.ok(surface);
  const sections = referenceSidebarSections(
    surface,
    [
      {
        type: 'group',
        label: 'Models',
        entries: [
          {
            type: 'link',
            label: 'ModelConfig',
            href: '/reference/modelconfig',
            isCurrent: true,
          },
        ],
      },
      {
        type: 'link',
        label: 'SDK',
        href: '/reference/sdk',
        isCurrent: false,
      },
      {
        type: 'link',
        label: 'TrainConfig',
        href: '/reference/trainconfig',
        isCurrent: false,
      },
    ],
    '/reference/modelconfig',
  );
  assert.equal(sections.length, 1);
  assert.equal(sections[0].label, 'SDK');
  assert.equal(sections[0].href, '/reference/sdk');
  assert.equal(sections[0].isCurrent, false);
  assert.deepEqual(
    sections[0].items.map((item) => item.label),
    ['ModelConfig', 'TrainConfig'],
  );
});
