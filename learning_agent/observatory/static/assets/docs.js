// Docs sidebar: one nav tree, injected into every docs page; the active page
// expands its in-page h2 anchors as a subtree. Also writes the prev/next
// footer so page order lives in exactly one place.
const NAV = [
  { href: "/docs",           title: "Introduction" },
  { href: "/docs/workflow",  title: "The Workflow" },
  { href: "/docs/tasks",     title: "Tasks & Tracks" },
  { href: "/docs/agent",     title: "The Learning Agent" },
  { href: "/docs/toolbox",   title: "The Learning Toolbox" },
  { href: "/docs/training",  title: "Training Pipeline" },
  { href: "/docs/eval",      title: "Evaluation & Scoring" },
  { href: "/docs/integrity", title: "Integrity Model" },
  { href: "/docs/runs",      title: "Runs & Observability" },
];

function currentPath() {
  let p = location.pathname.replace(/\/+$/, "");
  if (p === "" || p === "/how" || p === "/how.html" || p === "/docs.html") p = "/docs";
  return p;
}

function build() {
  const nav = document.getElementById("docs-nav");
  if (!nav) return;
  const here = currentPath();
  const title = document.createElement("p");
  title.className = "nav-title";
  title.textContent = "Learning agent documentation";
  nav.appendChild(title);
  for (const item of NAV) {
    const a = document.createElement("a");
    a.href = item.href;
    a.textContent = item.title;
    if (item.href === here) a.className = "active";
    nav.appendChild(a);
    if (item.href === here) {
      const sub = document.createElement("div");
      sub.className = "sub";
      for (const h2 of document.querySelectorAll(".docs-page h2[id]")) {
        const s = document.createElement("a");
        s.href = `#${h2.id}`;
        s.textContent = h2.textContent;
        sub.appendChild(s);
      }
      if (sub.children.length) nav.appendChild(sub);
    }
  }
  const idx = NAV.findIndex((n) => n.href === here);
  const foot = document.querySelector(".docs-page");
  if (idx !== -1 && foot) {
    const bar = document.createElement("div");
    bar.className = "pagenav";
    const prev = idx > 0 ? NAV[idx - 1] : null;
    const next = idx < NAV.length - 1 ? NAV[idx + 1] : null;
    bar.innerHTML =
      (prev ? `<a href="${prev.href}">← ${prev.title}</a>` : "<span></span>") +
      (next ? `<a href="${next.href}">${next.title} →</a>` : "<span></span>");
    foot.appendChild(bar);
  }
}
document.addEventListener("DOMContentLoaded", build);
