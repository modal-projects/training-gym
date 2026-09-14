export function sidebarScrollKey(section: string): string {
  return `tg-sidebar-scroll:${section}`;
}

export function nextPaneScrollTop(
  paneScrollTop: number,
  paneHeight: number,
  itemOffsetTop: number,
  itemHeight: number,
): number {
  const viewEnd = paneScrollTop + paneHeight;
  const itemEnd = itemOffsetTop + itemHeight;
  if (itemOffsetTop >= paneScrollTop && itemEnd <= viewEnd) {
    return paneScrollTop;
  }
  if (itemOffsetTop < paneScrollTop) {
    return itemOffsetTop;
  }
  return itemEnd - paneHeight;
}
