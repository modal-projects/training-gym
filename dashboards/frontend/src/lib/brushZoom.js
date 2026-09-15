// Svelte action port of the Modal dashboard's chart `Brush.svelte`:
//
//   - click-and-drag along the x-axis highlights an interval and zooms to it
//   - vertical wheel / pinch zooms in or out around the pointer
//   - horizontal wheel pans
//
// The action is unit-agnostic: every callback receives the new x-domain as
// fractions of the node's width, where `[0, 1]` is the domain currently on
// screen. Fractions outside `[0, 1]` mean "zoom out". The host converts the
// fractions to its own x units (rollout ids, bin values, ...).
//
//   use:brushZoom={{ onChangeDomainX: ([f0, f1]) => ..., enabled: true }}
//
// The node must be `position: relative` (or otherwise positioned) so the
// highlight it appends can be laid over the plot.

const MIN_DRAG_PX = 5;

export function brushZoom(node, params = {}) {
  let options = { enabled: true, enableBrushZoom: true, enableWheelZoom: true, ...params };
  let dragStartPx = null;
  let dragEndPx = null;
  let didDrag = false;

  const area = document.createElement("div");
  area.className = "chart-brush-area";
  area.setAttribute("aria-hidden", "true");
  area.style.display = "none";
  node.appendChild(area);
  node.classList.add("chart-brush-host");

  const enabled = () => options.enabled !== false && typeof options.onChangeDomainX === "function";
  const brushEnabled = () => enabled() && options.enableBrushZoom !== false;
  const wheelEnabled = () => enabled() && options.enableWheelZoom !== false;

  // Pointer position in px from the node's left edge, valid for window-level
  // events fired while the pointer is outside the node.
  const getPx = (event) => {
    const rect = node.getBoundingClientRect();
    return { px: event.clientX - rect.left, width: rect.width };
  };

  function renderArea() {
    if (dragStartPx == null || dragEndPx == null) {
      area.style.display = "none";
      return;
    }
    const left = Math.min(dragStartPx, dragEndPx);
    const width = Math.abs(dragEndPx - dragStartPx);
    area.style.display = "block";
    area.style.left = `${left}px`;
    area.style.width = `${width}px`;
  }

  function handlePointerDown(event) {
    didDrag = false;
    if (!brushEnabled() || event.button !== 0) return;
    const { px } = getPx(event);
    dragStartPx = px;
    dragEndPx = px;
    renderArea();
  }

  function handleWindowPointerMove(event) {
    if (dragStartPx == null) return;
    const { px, width } = getPx(event);
    dragEndPx = Math.min(Math.max(px, 0), width);
    renderArea();
  }

  function handlePointerUp() {
    if (dragStartPx != null && dragEndPx != null) {
      const width = node.getBoundingClientRect().width;
      if (width > 0 && Math.abs(dragEndPx - dragStartPx) >= MIN_DRAG_PX) {
        didDrag = true;
        const f0 = Math.min(dragStartPx, dragEndPx) / width;
        const f1 = Math.max(dragStartPx, dragEndPx) / width;
        options.onChangeDomainX([f0, f1]);
      }
    }
    dragStartPx = dragEndPx = null;
    renderArea();
  }

  // Swallow the click that trails a completed drag so host click handlers
  // don't also fire.
  function handleClick(event) {
    if (didDrag) {
      event.stopPropagation();
      event.preventDefault();
      didDrag = false;
    }
  }

  function handleWheel(event) {
    if (!wheelEnabled()) return;
    event.preventDefault();
    if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      const { width } = getPx(event);
      if (!width) return;
      const shift = event.deltaX / width;
      options.onChangeDomainX([shift, 1 + shift]);
    } else if (event.deltaY > 0.5 || event.deltaY < -0.5) {
      const factor = Math.exp(event.deltaY / 100);
      const { px, width } = getPx(event);
      if (!width) return;
      const center = Math.min(Math.max(px / width, 0), 1);
      options.onChangeDomainX([center - center * factor, center + (1 - center) * factor]);
    }
  }

  function syncCursor() {
    node.classList.toggle("chart-brush-enabled", brushEnabled());
  }

  node.addEventListener("pointerdown", handlePointerDown);
  node.addEventListener("click", handleClick);
  node.addEventListener("wheel", handleWheel, { passive: false });
  window.addEventListener("pointermove", handleWindowPointerMove);
  window.addEventListener("pointerup", handlePointerUp);
  window.addEventListener("pointercancel", handlePointerUp);
  syncCursor();

  return {
    update(next = {}) {
      options = { enabled: true, enableBrushZoom: true, enableWheelZoom: true, ...next };
      syncCursor();
    },
    destroy() {
      node.removeEventListener("pointerdown", handlePointerDown);
      node.removeEventListener("click", handleClick);
      node.removeEventListener("wheel", handleWheel);
      window.removeEventListener("pointermove", handleWindowPointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
      node.classList.remove("chart-brush-host", "chart-brush-enabled");
      area.remove();
    },
  };
}

/** Map fractional domain output from `brushZoom` onto a numeric `[min, max]` domain. */
export function fractionsToDomain([f0, f1], [xMin, xMax]) {
  const span = xMax - xMin;
  return [xMin + f0 * span, xMin + f1 * span];
}
