import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';

export interface TutorialSource {
  path: string;
  slug: string;
  runTarget: string;
  sourcePath: string;
}

const FOLDER_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

function assertUniqueTutorialSlugs(entries: TutorialSource[]): void {
  const pathsBySlug = new Map<string, string>();
  for (const entry of entries) {
    const previous = pathsBySlug.get(entry.slug);
    if (previous) {
      throw new Error(
        `Tutorial slug ${JSON.stringify(entry.slug)} is defined by both ${previous} and ${entry.path}`,
      );
    }
    pathsBySlug.set(entry.slug, entry.path);
  }
}

export async function discoverTutorialEntries(
  tutorialsDirectory: string,
): Promise<TutorialSource[]> {
  const children = (await readdir(tutorialsDirectory, { withFileTypes: true })).sort((left, right) =>
    left.name.localeCompare(right.name),
  );
  const entries: TutorialSource[] = [];
  for (const child of children) {
    if (child.isFile() && child.name.endsWith('.py')) {
      entries.push({
        path: path.join(tutorialsDirectory, child.name),
        slug: path.basename(child.name, '.py'),
        runTarget: `tutorials/${child.name}`,
        sourcePath: `tutorials/${child.name}`,
      });
      continue;
    }
    if (!child.isDirectory()) {
      continue;
    }
    const candidates: string[] = [];
    for (const filename of ['main.py', 'train.py']) {
      const candidate = path.join(tutorialsDirectory, child.name, filename);
      try {
        if ((await stat(candidate)).isFile()) candidates.push(candidate);
      } catch {
        // A folder may contain only helper files.
      }
    }
    if (candidates.length === 0) continue;
    if (candidates.length > 1) {
      throw new Error(`Tutorial folder ${JSON.stringify(child.name)} has multiple entrypoints`);
    }
    const tutorialPath = candidates[0];
    if (!FOLDER_NAME_PATTERN.test(child.name)) {
      throw new Error(
        `Tutorial folder ${JSON.stringify(child.name)} is not a valid Python module name; use only letters, digits, and underscores`,
      );
    }
    entries.push({
      path: tutorialPath,
      slug: child.name,
      runTarget: `-m tutorials.${child.name}.${path.basename(tutorialPath, '.py')}`,
      sourcePath: `tutorials/${child.name}`,
    });
  }
  assertUniqueTutorialSlugs(entries);
  return entries;
}
