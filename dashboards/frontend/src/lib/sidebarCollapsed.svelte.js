const STORAGE_KEY = "sidebarCollapsed";

function readStored() {
  if (typeof localStorage === "undefined") return undefined;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw === "true" || raw === "false") return raw === "true";
  return undefined;
}

// Reactive collapsed flag for the primary sidebar, persisted so the layout
// comes back the way it was left across reloads.
export function createSidebarCollapsedState(initial = false) {
  let collapsed = $state(readStored() ?? initial);

  return {
    get collapsed() {
      return collapsed;
    },
    set collapsed(value) {
      collapsed = value;
      if (typeof localStorage !== "undefined") {
        localStorage.setItem(STORAGE_KEY, String(value));
      }
    },
    toggle() {
      this.collapsed = !collapsed;
    },
  };
}
