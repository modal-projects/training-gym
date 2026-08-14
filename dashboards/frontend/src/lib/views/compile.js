const compiledModules = new Map();

async function sourceHash(source) {
  const bytes = new TextEncoder().encode(source);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

function rewriteImports(code) {
  return code
    .replace(
      /from\s+["'](svelte\/internal\/client|svelte|\$host\/[^"']+)["']/g,
      'from "__host:$1"',
    )
    .replace(/export\s+default\s+/g, "return ");
}

export async function compileAuthoredView(source) {
  const hash = await sourceHash(source);
  if (compiledModules.has(hash)) return compiledModules.get(hash);
  const { compile } = await import("svelte/compiler");
  const result = compile(source, {
    generate: "client",
    runes: true,
    name: "AuthoredView",
  });
  const body = rewriteImports(result.js.code);
  const factory = new Function("__require", body);
  const module = factory((specifier) => {
    if (!specifier.startsWith("__host:")) {
      throw new Error(`Import not allowed: ${specifier}`);
    }
    return globalThis.__TRAINING_GYM_HOST_REGISTRY?.[specifier.slice("__host:".length)];
  });
  compiledModules.set(hash, module);
  return module;
}
