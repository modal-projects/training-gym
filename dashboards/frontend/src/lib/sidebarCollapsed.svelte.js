const STORAGE_KEY = "sidebarCollapsed";

// Web Storage can be present but throw (storage blocked by browser policy),
// so a failed read is treated as no stored preference.
function readStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === "true" || raw === "false") return raw === "true";
  } catch {
    // fall through
  }
  return undefined;
}

function writeStored(value) {
  try {
    localStorage.setItem(STORAGE_KEY, String(value));
  } catch {
    // in-memory state still applies for this page load
  }
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
      writeStored(value);
    },
    toggle() {
      this.collapsed = !collapsed;
    },
  };
}
