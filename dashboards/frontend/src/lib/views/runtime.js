import { builtinViewDocs, builtinLayouts, resolveLayout, resolveView } from "../../views/registry.js";

export { builtinViewDocs, builtinLayouts, resolveLayout, resolveView };

export function safeMode() {
  return typeof window !== "undefined" && new URLSearchParams(window.location.search).get("safe") === "1";
}
