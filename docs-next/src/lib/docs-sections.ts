export const DOCS_NAV = [
  { section: 'Guides', prefix: '/guides' },
  { section: 'Tutorials', prefix: '/tutorials' },
  { section: 'Reference', prefix: '/reference' },
] as const;

export type DocsSection = (typeof DOCS_NAV)[number]['section'];

const PREFIXES_LONGEST_FIRST = [...DOCS_NAV].sort(
  (a, b) => b.prefix.length - a.prefix.length,
);

export function sectionForUrl(url: string): DocsSection | undefined {
  const pathname = url.split(/[?#]/)[0] ?? url;
  const path =
    pathname.length > 1 && pathname.endsWith('/') ? pathname.slice(0, -1) : pathname;
  for (const { section, prefix } of PREFIXES_LONGEST_FIRST) {
    if (path === prefix || path.startsWith(`${prefix}/`)) return section;
  }
  return undefined;
}
