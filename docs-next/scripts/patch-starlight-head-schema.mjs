import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const targetPath = resolve(
  scriptDir,
  '..',
  'node_modules',
  '@astrojs',
  'starlight',
  'schemas',
  'head.ts',
);

const source = readFileSync(targetPath, 'utf8');
const original = "attrs: z.record(z.union([z.string(), z.boolean(), z.undefined()])).optional(),";
const replacement =
  "attrs: z.record(z.string(), z.union([z.string(), z.boolean(), z.undefined()])).optional(),";

if (source.includes(replacement)) {
  process.exit(0);
}

if (!source.includes(original)) {
  throw new Error(`Could not patch ${targetPath}: expected schema not found.`);
}

writeFileSync(targetPath, source.replace(original, replacement));
