import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compile } from "svelte/compiler";
import * as esbuild from "esbuild";

const [, , sourcePath, outputPath] = process.argv;
if (!sourcePath || !outputPath) {
  console.error("usage: node component_bundle.mjs SOURCE OUTPUT");
  process.exit(2);
}

const source = await fs.readFile(sourcePath, "utf8");
const compiled = compile(source, {
  generate: "client",
  css: "injected",
  dev: false,
});

const temporary = `${outputPath}.component.js`;
const frontendRoot = path.dirname(fileURLToPath(import.meta.url));
await fs.writeFile(temporary, compiled.js.code, "utf8");
try {
  const entry = [
    `import Component from ${JSON.stringify(temporary)};`,
    `import { mount } from "svelte";`,
    `export function mountViewer(target, props) {`,
    `  return mount(Component, { target, props });`,
    `}`,
  ].join("\n");
  const result = await esbuild.build({
    stdin: { contents: entry, resolveDir: path.dirname(temporary), sourcefile: "entry.js" },
    bundle: true,
    format: "esm",
    platform: "browser",
    target: "es2020",
    nodePaths: [path.join(frontendRoot, "node_modules")],
    write: false,
    sourcemap: false,
  });
  await fs.writeFile(outputPath, result.outputFiles[0].contents);
} finally {
  await fs.rm(temporary, { force: true });
}
