<script>
  // Loading placeholder for the graph components on the run detail page. Mirrors
  // the footprint of the real chart (optional title, a plot box the same height,
  // and mark shapes appropriate to the chart type) so the layout doesn't jump
  // when data arrives.
  //
  //   variant   "line" | "bars" | "violins"  — which chart it stands in for
  //   height    number (px plot height)
  //   count     number of marks (bars / violins)
  //   showTitle render a title pulse above the plot (charts whose title is drawn
  //             by the chart component itself, e.g. the reward LineChart)

  let {
    variant = "bars",
    height = 150,
    count = 12,
    showTitle = false,
  } = $props();

  // Deterministic mark heights (no Math.random — keeps SSR/hydration stable and
  // avoids the reroll-on-every-render flicker). A blend of two sines reads as an
  // irregular distribution rather than an obvious repeating wave.
  let marks = $derived(
    Array.from({ length: Math.max(1, count) }, (_, i) => {
      const t = Math.abs(Math.sin(i * 1.3) * 0.6 + Math.sin(i * 0.7 + 1) * 0.4);
      return 22 + Math.round(t * 66); // 22%–88% of plot height
    }),
  );
</script>

<div class="w-full" aria-hidden="true">
  {#if showTitle}
    <span class="cs-pulse w-[40%] max-w-[160px] h-[12px] mb-[8px]"></span>
  {/if}
  <div class="cs-plot" class:bordered={variant !== "line"} style:height={`${height}px`}>
    {#if variant === "line"}
      <span class="cs-grid" style:top="25%"></span>
      <span class="cs-grid" style:top="50%"></span>
      <span class="cs-grid" style:top="75%"></span>
      <span class="cs-pulse cs-area"></span>
    {:else}
      <div class="cs-marks">
        {#each marks as h, i (i)}
          <span
            class="cs-pulse"
            class:cs-bar={variant === "bars"}
            class:cs-violin={variant === "violins"}
            style:height={`${h}%`}
          ></span>
        {/each}
      </div>
    {/if}
  </div>
</div>
