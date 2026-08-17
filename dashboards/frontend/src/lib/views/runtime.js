export function safeMode() {
  return typeof window !== "undefined" && new URLSearchParams(window.location.search).get("safe") === "1";
}
