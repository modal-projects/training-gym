// Minimal ANSI escape-sequence parser for the live log viewer.
//
// Modal/Ray/Megatron logs are emitted with raw ANSI SGR ("Select Graphic
// Rendition") codes for color and weight. The backend streams log lines
// verbatim, so by the time a line reaches the browser it still carries
// sequences like "\x1b[32mINFO\x1b[0m". Rendered as plain text these show up
// as garbage ("[32mINFO[0m"), so we parse them here into styled segments.
//
// parseAnsi(line) -> [{ text, style }]
//   - `style` is an inline CSS string (possibly empty) for a <span>.
//   - Non-SGR escape sequences (cursor moves, clears, etc.) and stray C0
//     control characters are dropped so they don't corrupt the output.

// VS Code's default dark-terminal palette — reads well on the log panel's
// dark background. Indices 0-7 normal, 8-15 bright.
const BASE_COLORS = [
  "#000000", "#cd3131", "#0dbc79", "#e5e510",
  "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
  "#666666", "#f14c4c", "#23d18b", "#f5f543",
  "#3b8eea", "#d670d6", "#29b8db", "#ffffff",
];

// Resolve an xterm-256 color index to a hex string.
function color256(n) {
  if (n < 16) return BASE_COLORS[n];
  if (n < 232) {
    // 6x6x6 color cube.
    const i = n - 16;
    const r = Math.floor(i / 36);
    const g = Math.floor((i % 36) / 6);
    const b = i % 6;
    const c = (v) => (v === 0 ? 0 : v * 40 + 55);
    const hex = (v) => c(v).toString(16).padStart(2, "0");
    return `#${hex(r)}${hex(g)}${hex(b)}`;
  }
  // 24-step grayscale ramp.
  const v = (n - 232) * 10 + 8;
  const hex = v.toString(16).padStart(2, "0");
  return `#${hex}${hex}${hex}`;
}

// Mutating state -> inline CSS for the current run of text.
function styleFor(state) {
  const parts = [];
  let fg = state.fg;
  if (state.bold) parts.push("font-weight:600");
  if (state.dim) parts.push("opacity:0.65");
  if (state.italic) parts.push("font-style:italic");
  if (state.underline) parts.push("text-decoration:underline");
  if (state.inverse) {
    [fg, state.bg] = [state.bg ?? "#0d0d0d", fg ?? "#e5e5e5"];
  }
  if (fg) parts.push(`color:${fg}`);
  if (state.bg) parts.push(`background-color:${state.bg}`);
  return parts.join(";");
}

// Apply one SGR parameter list (the numbers in "\x1b[ ... m").
function applySgr(params, state) {
  for (let i = 0; i < params.length; i++) {
    const code = params[i];
    if (code === 0 || Number.isNaN(code)) {
      state.fg = null;
      state.bg = null;
      state.bold = state.dim = state.italic = state.underline = state.inverse = false;
    } else if (code === 1) state.bold = true;
    else if (code === 2) state.dim = true;
    else if (code === 3) state.italic = true;
    else if (code === 4) state.underline = true;
    else if (code === 7) state.inverse = true;
    else if (code === 22) state.bold = state.dim = false;
    else if (code === 23) state.italic = false;
    else if (code === 24) state.underline = false;
    else if (code === 27) state.inverse = false;
    else if (code >= 30 && code <= 37) state.fg = BASE_COLORS[code - 30];
    else if (code >= 90 && code <= 97) state.fg = BASE_COLORS[code - 90 + 8];
    else if (code >= 40 && code <= 47) state.bg = BASE_COLORS[code - 40];
    else if (code >= 100 && code <= 107) state.bg = BASE_COLORS[code - 100 + 8];
    else if (code === 39) state.fg = null;
    else if (code === 49) state.bg = null;
    else if (code === 38 || code === 48) {
      // Extended color: "38;5;n" (256) or "38;2;r;g;b" (truecolor).
      const target = code === 38 ? "fg" : "bg";
      const mode = params[i + 1];
      if (mode === 5) {
        state[target] = color256(params[i + 2] || 0);
        i += 2;
      } else if (mode === 2) {
        const r = params[i + 2] || 0;
        const g = params[i + 3] || 0;
        const b = params[i + 4] || 0;
        state[target] = `rgb(${r},${g},${b})`;
        i += 4;
      }
    }
  }
}

// eslint-disable-next-line no-control-regex
const ESC = /\x1b\[([0-9;]*)([A-Za-z])/g;
// eslint-disable-next-line no-control-regex
const STRAY_CONTROL = /[\x00-\x08\x0b-\x1f\x7f]/g;

export function parseAnsi(line) {
  if (typeof line !== "string" || line.length === 0) {
    return [{ text: "", style: "" }];
  }
  // Fast path: no escape character at all (the common case).
  if (line.indexOf("\x1b") === -1) {
    return [{ text: line.replace(STRAY_CONTROL, ""), style: "" }];
  }

  const segments = [];
  const state = {
    fg: null, bg: null,
    bold: false, dim: false, italic: false, underline: false, inverse: false,
  };
  let style = "";
  let lastIndex = 0;
  let match;
  ESC.lastIndex = 0;

  const push = (text) => {
    if (!text) return;
    const clean = text.replace(STRAY_CONTROL, "");
    if (clean) segments.push({ text: clean, style });
  };

  while ((match = ESC.exec(line)) !== null) {
    push(line.slice(lastIndex, match.index));
    lastIndex = ESC.lastIndex;
    // Only SGR ("m") sequences affect styling; everything else is dropped.
    if (match[2] === "m") {
      const params = match[1] === "" ? [0] : match[1].split(";").map((n) => parseInt(n, 10));
      applySgr(params, state);
      style = styleFor(state);
    }
  }
  push(line.slice(lastIndex));

  return segments.length ? segments : [{ text: "", style: "" }];
}
