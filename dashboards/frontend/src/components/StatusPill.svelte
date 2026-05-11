<script>
  import { CheckCircle2, CircleX, Loader2, OctagonX } from "lucide-svelte";

  let { status, iconOnly = false } = $props();

  const STATUS_MAP = {
    completed: "Completed",
    pending: "Pending",
    running: "Pending",
    stopped: "Stopped",
    failed: "Failed",
  };

  let normalizedStatus = $derived.by(() => {
    const s = String(status || "").toLowerCase();
    if (s === "running") return "pending";
    return s in STATUS_MAP ? s : "pending";
  });

  let label = $derived(STATUS_MAP[normalizedStatus] ?? "Running");
</script>

<div
  class="status-pill"
  class:icon-only={iconOnly}
  class:status-completed={normalizedStatus === "completed"}
  class:status-running={normalizedStatus === "running"}
  class:status-pending={normalizedStatus === "pending"}
  class:status-stopped={normalizedStatus === "stopped"}
  class:status-failed={normalizedStatus === "failed"}
  aria-label={label}
>
  {#if normalizedStatus === "completed"}
    <CheckCircle2 size={14} />
  {:else if normalizedStatus === "stopped"}
    <OctagonX size={14} />
  {:else if normalizedStatus === "failed"}
    <CircleX size={14} />
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
    gap: 6px;
    white-space: nowrap;
    border-radius: 9999px;
    padding: 3px 10px;
    font-size: 12px;
    line-height: 12px;
    border: 1px solid transparent;
    box-sizing: border-box;
    width: fit-content;
  }

  .status-pill.icon-only {
    min-width: 24px;
    width: 24px;
    padding-inline: 0;
    gap: 0;
    justify-content: center;
  }

  .status-running {
    background: #2f2436;
    color: #d176bd;
    border-color: #3b2a37;
  }

  .status-pending {
    background: #2f2436;
    color: #d176bd;
    border-color: #3b2a37;
  }

  .status-completed {
    background: #273823;
    color: #6ac355;
    border-color: #2d4327;
  }

  .status-stopped {
    background: #3b2f20;
    color: #ffab5e;
    border-color: #5d442d;
  }

  .status-failed {
    background: #3b2020;
    color: #f87171;
    border-color: #5b3333;
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
