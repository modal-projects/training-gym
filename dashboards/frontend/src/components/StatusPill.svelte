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
    justify-content: center;
    gap: 0.35rem;
    white-space: nowrap;
    border-radius: 9999px;
    padding: 0.2rem 0.65rem;
    min-width: 96px;
    min-height: 24px;
    font-size: 0.72rem;
    line-height: 1;
    border: 1px solid transparent;
    box-sizing: border-box;
  }

  .status-pill.icon-only {
    min-width: 24px;
    width: 24px;
    padding-inline: 0;
    gap: 0;
  }

  .status-running {
    background: #22313f;
    color: #91c8ef;
    border-color: #303b43;
  }

  .status-pending {
    background: #2f2436;
    color: #ff8de6;
    border-color: #3b2a37;
  }

  .status-completed {
    background: #273823;
    color: #7fee64;
    border-color: #2d4327;
  }

  .status-stopped {
    background: #3b2f20;
    color: #fb923c;
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
