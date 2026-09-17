// Single-key global hotkeys, after the Modal dashboard's `useGlobalHotkey`.
// Keys typed into inputs, textareas, selects and editable regions are left
// alone, as are chords with a modifier held.

function isTextInput(target) {
  if (!target || !(target instanceof Element)) return false;
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target.isContentEditable
  );
}

/**
 * Bind `callback` to `key` for the lifetime of the calling component.
 * `enabled` is re-read on every keypress so callers can pass a getter.
 */
export function useGlobalHotkey(key, callback, enabled = () => true) {
  $effect(() => {
    if (typeof window === "undefined") return;
    const handler = (event) => {
      if (event.defaultPrevented || event.repeat) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key !== key) return;
      if (isTextInput(event.target)) return;
      if (!enabled()) return;
      event.preventDefault();
      callback();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });
}
