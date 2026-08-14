import * as svelte from "svelte";
import * as svelteInternalClient from "svelte/internal/client";
import * as hostComponents from "$host/components";
import * as hostFormat from "$host/format";
import * as hostData from "$host/data";
import * as hostIcons from "$host/icons";

const compiledModules = new Map();
const injectedStyles = new Set();

const allowedModules = {
  svelte,
  "svelte/internal/client": svelteInternalClient,
  "$host/components": hostComponents,
  "$host/format": hostFormat,
  "$host/data": hostData,
  "$host/icons": hostIcons,
};

async function sourceHash(source) {
  const bytes = new TextEncoder().encode(source);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

function rewriteImports(code) {
  code = code.replace(
    /(^|\n)\s*import\s*(['"])([^'"]+)\2\s*;?/g,
    (_, prefix, _quote, specifier) => {
      if (specifier !== "svelte/internal/disclose-version") {
        throw new Error(`Import not allowed: ${specifier}`);
      }
      return prefix;
    },
  );
  const importPattern =
    /(^|\n)\s*import\s+([\s\S]*?)\s+from\s+(['"])([^'"]+)\3\s*;?/g;
  code = code.replace(importPattern, (_, prefix, clause, _quote, specifier) => {
    if (!Object.hasOwn(allowedModules, specifier)) {
      throw new Error(`Import not allowed: ${specifier}`);
    }
    const moduleExpression = `__require(${JSON.stringify(specifier)})`;
    const trimmed = clause.trim();
    if (trimmed.startsWith("* as ")) {
      return `${prefix}const ${trimmed.slice(5).trim()} = ${moduleExpression};`;
    }
    const bindings = [];
    function addNamedBindings(namedClause) {
      for (const entry of namedClause
        .replace(/^\{\s*|\s*\}$/g, "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)) {
        const [imported, local] = entry.split(/\s+as\s+/);
        const property = imported.trim();
        const binding = (local || imported).trim();
        bindings.push(`${property}: ${binding}`);
      }
    }
    if (trimmed.startsWith("{")) {
      addNamedBindings(trimmed);
    } else {
      const comma = trimmed.indexOf(",");
      const defaultName = (comma < 0 ? trimmed : trimmed.slice(0, comma)).trim();
      bindings.push(`default: ${defaultName}`);
      if (comma >= 0) addNamedBindings(trimmed.slice(comma + 1).trim());
    }
    return `${prefix}const { ${bindings.join(", ")} } = ${moduleExpression};`;
  });
  return code.replace(/export\s+default\s+/g, "return ");
}

function injectStyles(hash, css) {
  if (!css || typeof document === "undefined" || injectedStyles.has(hash)) {
    return;
  }
  const style = document.querySelector(`style[data-authored-view="${hash}"]`);
  if (style) {
    injectedStyles.add(hash);
    return;
  }
  const authoredStyle = document.createElement("style");
  authoredStyle.dataset.authoredView = hash;
  authoredStyle.textContent = css;
  document.head.appendChild(authoredStyle);
  injectedStyles.add(hash);
}

export async function compileAuthoredView(source) {
  const hash = await sourceHash(source);
  if (compiledModules.has(hash)) return compiledModules.get(hash);
  const compiler = await import("svelte/compiler");
  const { compile } = compiler.default || compiler;
  const result = compile(source, {
    generate: "client",
    runes: true,
    name: "AuthoredView",
  });
  injectStyles(hash, result.css?.code);
  const body = rewriteImports(result.js.code);
  const factory = new Function("__require", body);
  const module = factory((specifier) => {
    if (!Object.hasOwn(allowedModules, specifier)) {
      throw new Error(`Import not allowed: ${specifier}`);
    }
    return allowedModules[specifier];
  });
  compiledModules.set(hash, module);
  return module;
}
