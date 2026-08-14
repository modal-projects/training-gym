// Tools catalog: renders /assets/tools.json (emitted by observatory/validate_tools.py
// --emit-json). Hash routing: /tools#<category>/<name> selects a tool, so run-page
// timeline chips can deep-link. READMEs are rendered with a deliberately small
// markdown subset (headings, fences, inline code, bold, lists) — no external libs.
const $ = (s) => document.querySelector(s);
let CATALOG = null;

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function mdToHtml(md) {
  const lines = md.split("\n");
  const out = [];
  let inCode = false, inList = false;
  const closeList = () => { if (inList) { out.push("</ul>"); inList = false; } };
  for (const raw of lines) {
    if (raw.startsWith("```")) {
      closeList();
      out.push(inCode ? "</code></pre>" : "<pre><code>");
      inCode = !inCode;
      continue;
    }
    if (inCode) { out.push(esc(raw)); continue; }
    let line = esc(raw)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    if (/^# /.test(line)) { closeList(); out.push(`<h2>${line.slice(2)}</h2>`); }
    else if (/^## /.test(line)) { closeList(); out.push(`<h3>${line.slice(3)}</h3>`); }
    else if (/^- /.test(line)) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${line.slice(2)}</li>`);
    } else if (line.trim() === "") { closeList(); out.push(""); }
    else { closeList(); out.push(`<p>${line}</p>`); }
  }
  closeList();
  if (inCode) out.push("</code></pre>");
  return out.join("\n");
}

function toolId(t) { return `${t.category}/${t.name}`; }

function renderTree() {
  const tree = $("#tool-tree");
  tree.replaceChildren();
  const here = decodeURIComponent(location.hash.slice(1));
  for (const [cat, tools] of Object.entries(CATALOG.categories)) {
    const h = document.createElement("p");
    h.className = "cat";
    h.textContent = cat;
    tree.appendChild(h);
    for (const t of tools) {
      const a = document.createElement("a");
      a.href = `#${toolId(t)}`;
      if (toolId(t) === here) a.className = "active";
      const name = document.createElement("span");
      name.textContent = t.name;
      const kind = document.createElement("span");
      kind.className = "kind";
      kind.textContent = t.kind;
      a.append(name, kind);
      tree.appendChild(a);
    }
  }
  if ((CATALOG.packages || []).length) {
    const h = document.createElement("p");
    h.className = "cat";
    h.textContent = "cloned packages";
    tree.appendChild(h);
  }
  for (const pkg of CATALOG.packages || []) {
    const a = document.createElement("a");
    a.href = `#pkg/${pkg.name}`;
    if (`pkg/${pkg.name}` === here) a.className = "active";
    const name = document.createElement("span");
    name.textContent = pkg.name;
    const kind = document.createElement("span");
    kind.className = "kind";
    kind.textContent = "package";
    a.append(name, kind);
    tree.appendChild(a);
  }
}

function renderDetail() {
  const here = decodeURIComponent(location.hash.slice(1));
  if (here.startsWith("pkg/")) {
    const pkg = (CATALOG.packages || []).find((p) => p.name === here.slice(4));
    if (!pkg) return;
    const main = $("#tool-detail");
    main.replaceChildren();
    const head = document.createElement("div");
    head.className = "tool-head";
    head.innerHTML = `<h1>${esc(pkg.name)}</h1><span class="chip chip-kind">package</span>`;
    main.appendChild(head);
    const body = document.createElement("div");
    body.innerHTML =
      `<p class="tool-path">${esc(pkg.dest)}</p>` +
      `<p>${esc(pkg.notes || "")}</p>` +
      `<p><strong>upstream</strong> <a href="${esc(pkg.repo)}">${esc(pkg.repo)}</a> ` +
      `@ <code>${esc((pkg.commit || "").slice(0, 12))}</code></p>`;
    main.appendChild(body);
    renderTree();
    return;
  }
  const all = Object.values(CATALOG.categories).flat();
  // accept "<category>/<name>" and bare "<name>" (run-page timeline links)
  const t = all.find((x) => toolId(x) === here) || all.find((x) => x.name === here);
  if (t && toolId(t) !== here) history.replaceState(null, "", `#${toolId(t)}`);
  const main = $("#tool-detail");
  if (!t) return;
  main.replaceChildren();
  const head = document.createElement("div");
  head.className = "tool-head";
  head.innerHTML =
    `<h1>${esc(t.name)}</h1>` +
    `<span class="chip chip-kind">${esc(t.kind)}</span>` +
    (t.cost ? `<span class="chip chip-cost">${esc(t.cost)}</span>` : "") +
    (t.provenance ? `<span class="chip ${t.provenance === "invented" ? "chip-invented" : "chip-seed"}">` +
      `${esc(t.provenance)}${t.created_by ? " · " + esc(t.created_by) : ""}</span>` : "");
  main.appendChild(head);
  const path = document.createElement("p");
  path.className = "tool-path";
  path.textContent = `python3 ${t.path}`;
  main.appendChild(path);
  const sum = document.createElement("p");
  sum.className = "lede";
  sum.textContent = t.summary;
  main.appendChild(sum);

  // the tool.md is THE doc; flags come from `run.py --help`, harvested at
  // catalog generation. Old catalogs (pre-2026-08-05) carried inputs/outputs/
  // args/readme instead — render whichever fields exist.
  const doc = document.createElement("div");
  doc.id = "tool-readme";
  doc.innerHTML = mdToHtml(t.doc || t.readme || "");
  main.appendChild(doc);

  if (t.help) {
    const help = document.createElement("pre");
    help.className = "tool-help";
    help.textContent = t.help;
    main.appendChild(help);
  } else if (t.args) {
    const wrap = document.createElement("div");
    wrap.className = "tblwrap";
    const rows = Object.entries(t.args || {})
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(String(v))}</td></tr>`).join("");
    wrap.innerHTML = `<table class="args-table"><tr><th>flag</th><th>meaning</th></tr>${rows}</table>`;
    main.appendChild(wrap);
  }
  renderTree();
}

async function boot() {
  const r = await fetch("/assets/tools.json");
  CATALOG = await r.json();
  renderTree();
  if (location.hash.length > 1) renderDetail();
  window.addEventListener("hashchange", renderDetail);
}
boot().catch((e) => { $("#tool-tree").textContent = `failed to load catalog: ${e}`; });
