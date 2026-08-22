export const TUTORIAL_ENTRY_PREFIX = 'tutorials/';

export function isTutorialPage(idOrPath: string): boolean {
  return idOrPath.replace(/^\/+/, '').startsWith(TUTORIAL_ENTRY_PREFIX);
}

export function tutorialGithubUrl(id: string): string {
  const slug = id.replace(/^\/+/, '').slice(TUTORIAL_ENTRY_PREFIX.length);
  return `https://github.com/modal-projects/training-gym/blob/main/${TUTORIAL_ENTRY_PREFIX}${slug}.py`;
}
