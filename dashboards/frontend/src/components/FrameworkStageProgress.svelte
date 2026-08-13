<script>
  let { progress = null, progressLabel = "", stageLabel = "", compact = false, active = true } = $props();

  function progressPercent(value) {
    if (!value?.total) return 0;
    const current = Number(value.current);
    const total = Number(value.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
      return 0;
    }
    return Math.max(0, Math.min(100, (current / total) * 100));
  }
</script>

<div class="framework-stage-progress" class:compact>
  {#if progress}
    <span class="progress-track" aria-hidden="true">
      <span
        class="block h-full min-w-[2px] [border-radius:inherit] bg-(--accent)"
        class:progress-fill-done={!active}
        style={`width: ${progressPercent(progress)}%`}
      ></span>
    </span>
  {/if}
  {#if progressLabel}
    <span class="text-(--muted) text-[11px] font-medium leading-[12px] [font-variant-numeric:tabular-nums] min-w-0 max-w-full overflow-hidden text-ellipsis whitespace-nowrap">{progressLabel}</span>
  {/if}
  {#if stageLabel}
    <span class="text-[color-mix(in_srgb,var(--accent)_42%,var(--muted)_58%)] text-[11px] font-medium leading-[12px] [font-variant-numeric:tabular-nums] tracking-[0] [text-shadow:0_0_10px_color-mix(in_srgb,var(--accent)_18%,transparent)] [animation:stage-label-flash_1.2s_ease-in-out_infinite] min-w-0 max-w-full overflow-hidden text-ellipsis whitespace-nowrap" class:stage-label-done={!active}>{stageLabel}</span>
  {/if}
</div>
