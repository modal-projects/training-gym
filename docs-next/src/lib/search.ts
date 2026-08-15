import { DOCS_NAV, sectionForUrl, type DocsSection } from './docs-sections';

export type SearchHit = {
  url: string;
  title: string;
  subtitle?: string;
  children?: SearchHit[];
};
export type SearchGroup = { section: DocsSection; hits: SearchHit[] };
const MAX_HITS_PER_GROUP = 7;

type PagefindFragment = {
  url?: unknown;
  meta?: { title?: unknown };
  sub_results?: unknown;
};

function asNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function toSearchHit(fragment: unknown): SearchHit | undefined {
  if (typeof fragment !== 'object' || fragment === null) return undefined;
  const { url, meta, sub_results } = fragment as PagefindFragment;
  if (typeof url !== 'string' || url.length === 0) return undefined;
  const title = meta && typeof meta === 'object' ? asNonEmptyString(meta.title) : undefined;
  if (!title) return undefined;
  const children: SearchHit[] = [];
  if (Array.isArray(sub_results)) {
    for (const sub of sub_results) {
      if (typeof sub !== 'object' || sub === null) continue;
      const headingUrl = asNonEmptyString((sub as { url?: unknown }).url);
      const headingTitle = asNonEmptyString((sub as { title?: unknown }).title);
      if (!headingUrl || !headingTitle) continue;
      if (headingUrl === url || headingTitle === title) continue;
      children.push({ url: headingUrl, title: headingTitle, subtitle: title });
    }
  }
  const hit: SearchHit = { url, title };
  if (children.length > 0) hit.children = children;
  return hit;
}

export function groupHits(hits: readonly SearchHit[]): SearchGroup[] {
  const buckets = new Map<DocsSection, SearchHit[]>();
  for (const hit of hits) {
    const section = sectionForUrl(hit.url);
    const list = buckets.get(section);
    if (list) list.push(hit);
    else buckets.set(section, [hit]);
  }
  return DOCS_NAV.flatMap(({ section }) => {
    const sectionHits = buckets.get(section);
    if (!sectionHits?.length) return [];
    const out: SearchHit[] = [];
    let used = 0;
    for (const hit of sectionHits) {
      if (used >= MAX_HITS_PER_GROUP) break;
      const children = hit.children ?? [];
      const room = MAX_HITS_PER_GROUP - used;
      if (room === 1 || children.length === 0) {
        const page: SearchHit = { url: hit.url, title: hit.title };
        if (hit.subtitle) page.subtitle = hit.subtitle;
        out.push(page);
        used += 1;
        continue;
      }
      const kept = children.slice(0, room - 1);
      const page: SearchHit = { url: hit.url, title: hit.title, children: kept };
      if (hit.subtitle) page.subtitle = hit.subtitle;
      out.push(page);
      used += 1 + kept.length;
    }
    return [{ section, hits: out }];
  });
}
