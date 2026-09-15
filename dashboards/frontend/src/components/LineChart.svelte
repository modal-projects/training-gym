<script>
  import { brushZoom, fractionsToDomain } from "../lib/brushZoom.js";

  let {
    title = "",
    data = [],
    height = 140,
    color = "var(--accent)",
    ariaLabel = title || "Line chart",
    formatX = (row) => String(row?.x ?? ""),
    formatY = (value) => String(value),
    // `[min, max]` in x units; defaults to the data extent.
    xDomain = null,
    // Called with `[min, max]` in x units when the user drags or wheels.
    onChangeDomainX = null,
  } = $props();

  let chartEl = $state(null);
  let hoveredIndex = $state(null);
  let pendingEvent = null;
  let frame = null;

  let allRows = $derived(
    (Array.isArray(data) ? data : [])
      .map((row, index) => ({
        ...row,
        index,
        x: Number(row?.x),
        y: Number(row?.y),
      }))
      .filter((row) => Number.isFinite(row.x) && Number.isFinite(row.y)),
  );

  let hasDomain = $derived(
    Array.isArray(xDomain) &&
      Number.isFinite(xDomain[0]) &&
      Number.isFinite(xDomain[1]) &&
      xDomain[1] > xDomain[0],
  );
  let xMin = $derived(
    hasDomain ? xDomain[0] : allRows.length ? Math.min(...allRows.map((row) => row.x)) : 0,
  );
  let xMax = $derived(
    hasDomain ? xDomain[1] : allRows.length ? Math.max(...allRows.map((row) => row.x)) : 1,
  );
  let xSpan = $derived(xMax - xMin || 1);

  // Rows inside the visible window drive the y scale and hover; neighbours
  // just outside it are kept so the line runs off the edge instead of
  // stopping short.
  let rows = $derived(
    hasDomain ? allRows.filter((row) => row.x >= xMin && row.x <= xMax) : allRows,
  );
  let drawnRows = $derived.by(() => {
    if (!hasDomain) return allRows;
    const first = allRows.findIndex((row) => row.x >= xMin);
    if (first === -1) return allRows.slice(-1);
    let last = allRows.length - 1;
    while (last > 0 && allRows[last].x > xMax) last--;
    return allRows.slice(Math.max(0, first - 1), Math.min(allRows.length, last + 2));
  });

  let yMin = $derived(rows.length ? Math.min(0, ...rows.map((row) => row.y)) : 0);
  let yMax = $derived(rows.length ? Math.max(0, ...rows.map((row) => row.y)) : 1);
  let ySpan = $derived(yMax - yMin || 1);

  let singlePoint = $derived(rows.length === 1 && !hasDomain);

  function point(row) {
    const x = singlePoint ? 2 : ((row.x - xMin) / xSpan) * 100;
    const y = singlePoint ? 50 : 100 - ((row.y - yMin) / ySpan) * 96 - 2;
    return { x, y };
  }

  let zoomable = $derived(typeof onChangeDomainX === "function");

  function handleBrush(fractions) {
    if (!zoomable) return;
    onChangeDomainX(fractionsToDomain(fractions, [xMin, xMax]));
  }

  let path = $derived(
    drawnRows
      .map((row, index) => {
        const p = point(row);
        return `${index === 0 ? "M" : "L"} ${p.x.toFixed(3)} ${p.y.toFixed(3)}`;
      })
      .join(" "),
  );
  let hoveredRow = $derived(
    hoveredIndex == null ? null : rows[Math.max(0, Math.min(rows.length - 1, hoveredIndex))],
  );
  let hoveredPoint = $derived(hoveredRow ? point(hoveredRow) : null);
  let reverseTooltip = $derived(hoveredPoint ? hoveredPoint.x > 72 : false);

  function updateHoverFromPointer(event) {
    if (!chartEl || !rows.length) return;
    const rect = chartEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    const ratio = rect.width ? x / rect.width : 0;
    const targetX = xMin + ratio * xSpan;
    let nearestIndex = 0;
    let nearestDistance = Infinity;
    rows.forEach((row, index) => {
      const distance = Math.abs(row.x - targetX);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });
    hoveredIndex = nearestIndex;
  }

  function onPointerMove(event) {
    pendingEvent = event;
    if (frame != null) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      if (pendingEvent) updateHoverFromPointer(pendingEvent);
    });
  }

  function onPointerLeave() {
    hoveredIndex = null;
    pendingEvent = null;
  }

  $effect(() => {
    return () => {
      if (frame != null) cancelAnimationFrame(frame);
    };
  });
</script>

<div class="min-w-0">
  {#if title}
    <div class="text-(--text-bright) text-[12px] font-[600] mb-[6px]">{title}</div>
  {/if}

  {#if rows.length || (hasDomain && allRows.length)}
    <div
      class="relative overflow-hidden bg-(--color-c-gray-08,#1c1c1c) rounded-[6px] cursor-crosshair"
      bind:this={chartEl}
      style:height={`${height}px`}
      role="img"
      aria-label={ariaLabel}
      onpointermove={onPointerMove}
      onpointerleave={onPointerLeave}
      use:brushZoom={{ onChangeDomainX: handleBrush, enabled: zoomable }}
    >
      <svg class="block w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <path d={path} fill="none" stroke={color} stroke-width="1.5" vector-effect="non-scaling-stroke" />
        {#if hoveredPoint}
          <line
            x1={hoveredPoint.x}
            x2={hoveredPoint.x}
            y1="0"
            y2="100"
            class="stroke-[rgba(255,255,255,0.22)] [stroke-width:1]"
            vector-effect="non-scaling-stroke"
          />
        {/if}
      </svg>

      {#if !rows.length}
        <div class="absolute inset-0 flex items-center justify-center text-(--muted) text-[12px] pointer-events-none">
          No data in this range.
        </div>
      {/if}

      {#if singlePoint}
        {@const p = point(rows[0])}
        <span
          class="point-dot"
          style:left={`${p.x}%`}
          style:top={`${p.y}%`}
          style:background={color}
        ></span>
      {/if}

      {#if hoveredPoint}
        <span
          class="point-dot z-[2]! w-[8px]! h-[8px]!"
          style:left={`${hoveredPoint.x}%`}
          style:top={`${hoveredPoint.y}%`}
          style:background={color}
        ></span>
      {/if}

      {#if hoveredRow && hoveredPoint}
        <div
          class="chart-tooltip"
          class:reverse={reverseTooltip}
          style:left={`${hoveredPoint.x}%`}
        >
          <div class="text-[rgba(255,255,255,0.72)]">{formatX(hoveredRow)}</div>
          <div class="[color:white] font-[600] [font-variant-numeric:tabular-nums]">{formatY(hoveredRow.y, hoveredRow)}</div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="text-(--muted) text-[12px] leading-[16px]">No data.</div>
  {/if}
</div>
