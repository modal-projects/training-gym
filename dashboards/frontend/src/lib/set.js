// Returns a new Set with `key` toggled — added if absent, removed if present.
// Reassign the result to a $state Set so Svelte picks up the change.
export function toggleInSet(set, key) {
  const next = new Set(set);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  return next;
}
