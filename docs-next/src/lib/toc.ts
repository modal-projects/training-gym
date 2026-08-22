export type TocItem = {
  slug: string;
  text: string;
  depth: number;
  children: readonly TocItem[];
};

export type TocNode = {
  slug: string;
  text: string;
  children: readonly TocNode[];
};

const PAGE_TITLE_SLUG = '_top';

function toNode(item: TocItem): TocNode {
  return {
    slug: item.slug,
    text: item.text,
    children: item.children.map(toNode),
  };
}

export function titledToc(items: readonly TocItem[], title: string): TocNode {
  const children: TocNode[] = [];
  for (const item of items) {
    if (item.slug === PAGE_TITLE_SLUG) {
      children.push(...item.children.map(toNode));
      continue;
    }
    children.push(toNode(item));
  }
  return { slug: PAGE_TITLE_SLUG, text: title, children };
}
