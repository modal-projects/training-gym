const TOC_LINK = '.tg-toc a';

interface Section {
  id: string;
  heading: HTMLElement;
}

function linkTarget(link: HTMLAnchorElement): string {
  return decodeURIComponent(link.hash.replace(/^#/, ''));
}

function markCurrent(ids: ReadonlySet<string>): void {
  for (const node of document.querySelectorAll(TOC_LINK)) {
    const link = node as HTMLAnchorElement;
    if (ids.has(linkTarget(link))) {
      link.setAttribute('aria-current', 'true');
    } else {
      link.removeAttribute('aria-current');
    }
  }
}

function start(): void {
  const links = [...document.querySelectorAll<HTMLAnchorElement>(TOC_LINK)];
  if (links.length === 0) return;

  const sections = [...new Set(links.map(linkTarget))].flatMap((id): Section[] => {
    const heading = document.getElementById(id);
    return heading ? [{ id, heading }] : [];
  });
  const article = document.querySelector<HTMLElement>('.sl-markdown-content');
  if (sections.length === 0 || !article) return;

  const update = () => {
    const viewportTop =
      (document.querySelector('header')?.getBoundingClientRect().height ?? 0) + 16;
    const viewportBottom = window.innerHeight - 16;
    const articleBottom = article.getBoundingClientRect().bottom;
    const visible = new Set<string>();

    sections.forEach(({ id, heading }, index) => {
      const top = heading.getBoundingClientRect().top;
      const bottom =
        sections[index + 1]?.heading.getBoundingClientRect().top ?? articleBottom;
      if (top < viewportBottom && bottom > viewportTop) visible.add(id);
    });

    markCurrent(visible);
  };

  let frame = 0;
  const scheduleUpdate = () => {
    if (frame) return;
    frame = window.requestAnimationFrame(() => {
      frame = 0;
      update();
    });
  };

  update();
  window.addEventListener('scroll', scheduleUpdate, { passive: true });
  window.addEventListener('resize', scheduleUpdate);
}

const onIdle =
  window.requestIdleCallback ?? ((cb: IdleRequestCallback) => window.setTimeout(cb, 1));
onIdle(() => start());
