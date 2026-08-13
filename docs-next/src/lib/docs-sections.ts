export const DOCS_NAV = [
  { section: 'Overview', prefix: '/' },
  { section: 'Tutorials', prefix: '/tutorials/' },
  { section: 'API Reference', prefix: '/reference/' },
  { section: 'Support', prefix: '/support/' },
] as const;

export type DocsSection = (typeof DOCS_NAV)[number]['section'];

const PREFIXES_LONGEST_FIRST = [...DOCS_NAV].sort(
  (a, b) => b.prefix.length - a.prefix.length,
);

export function sectionForUrl(url: string): DocsSection {
  const pathname = url.split(/[?#]/)[0] ?? url;
  const path = pathname.endsWith('/') ? pathname : `${pathname}/`;
  for (const { section, prefix } of PREFIXES_LONGEST_FIRST) {
    if (path === prefix || path.startsWith(prefix)) return section;
  }
  return 'Overview';
}
