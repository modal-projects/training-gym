<script>
  import { CheckCircle2, Loader2, OctagonX, TriangleAlert } from "lucide-svelte";

  let { status, iconOnly = false } = $props();

  const STATUS_MAP = {
    completed: "Completed",
    stopped: "Stopped",
    failed: "Failed",
  };

  let normalizedStatus = $derived.by(() => {
    const s = String(status || "").toLowerCase();
    return s in STATUS_MAP ? s : "running";
  });

  let label = $derived(STATUS_MAP[normalizedStatus] ?? "Running");
</script>

<div
  class="status-pill"
  class:status-completed={normalizedStatus === "completed"}
  class:status-running={normalizedStatus === "running"}
  class:status-stopped={normalizedStatus === "stopped"}
  class:status-failed={normalizedStatus === "failed"}
  aria-label={label}
>
  {#if normalizedStatus === "completed"}
    <CheckCircle2 size={14} />
  {:else if normalizedStatus === "stopped"}
    <OctagonX size={14} />
  {:else if normalizedStatus === "failed"}
    <TriangleAlert size={14} />
  {:else}
    <span class="live-spinner">
      <Loader2 size={16} />
    </span>
  {/if}
  {#if !iconOnly}
    <span class="status-text">{label}</span>
  {/if}
</div>

<style>
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    white-space: nowrap;
    border-radius: 9999px;
    padding: 0.2rem 0.5rem;
    font-size: 0.72rem;
    line-height: 1;
    border: 0;
  }

  .status-running {
    background: #2f2f2f;
    color: #a3a3a3;
  }

  .status-completed {
    background: #273823;
    color: #7fee64;
  }

  .status-stopped {
    background: #3b2f20;
    color: #fb923c;
  }

  .status-failed {
    background: #3b2020;
    color: #f87171;
  }

  .status-text {
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }

  .live-spinner {
    animation: status-pill-spin 1s linear infinite;
  }

  @keyframes status-pill-spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>
