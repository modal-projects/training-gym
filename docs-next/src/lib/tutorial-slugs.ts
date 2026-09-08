import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';

export interface TutorialSource {
  path: string;
  slug: string;
  runTarget: string;
  sourcePath: string;
}

export function assertUniqueTutorialSlugs(entries: TutorialSource[]): void {
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
    const tutorialPath = path.join(tutorialsDirectory, child.name, 'main.py');
    try {
      const info = await stat(tutorialPath);
      if (info.isFile()) {
        entries.push({
          path: tutorialPath,
          slug: child.name,
          runTarget: `-m tutorials.${child.name}.main`,
          sourcePath: `tutorials/${child.name}`,
        });
      }
    } catch {
      continue;
    }
  }
  assertUniqueTutorialSlugs(entries);
  return entries;
}
