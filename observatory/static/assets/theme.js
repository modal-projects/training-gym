// Theme bootstrap — classic script, loaded in <head> so the first paint is themed.
(function () {
  const KEY = "obs-theme";
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { /* private mode */ }
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = saved || (prefersDark ? "dark" : "light");

  window.obsToggleTheme = function () {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(KEY, next); } catch (e) { /* ignore */ }
    document.dispatchEvent(new CustomEvent("obs-themechange", { detail: next }));
  };

  document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.addEventListener("click", window.obsToggleTheme);
  });
})();
