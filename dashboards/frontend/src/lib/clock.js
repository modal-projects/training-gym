import { readable } from "svelte/store";

// One ticking clock for every relative timestamp on the page. A per-instance
// interval costs a timer and a re-render per rendered timestamp, and the run
// list renders hundreds of them.
export const nowMs = readable(Date.now(), (set) => {
  if (typeof window === "undefined") return;
  const id = window.setInterval(() => set(Date.now()), 5000);
  return () => window.clearInterval(id);
});
