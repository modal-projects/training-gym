<script>
  import { onMount } from "svelte";
  import { fetchRun, fetchRunLogs } from "$host/data";

  let { runId, initialRun = null } = $props();
  let run = $state(null);
  let lines = $state([]);
  let search = $state("");
  let loading = $state(true);
  let error = $state("");
  let streamState = $state("idle");
  let since = $state("");
  let until = $state("");

  $effect(() => {
    if (initialRun?.run_id === runId) run = initialRun;
  });

  let isRunning = $derived(String(run?.status || "").toLowerCase() === "running");

  function localRange(value) {
    if (!value) return "";
    const date = new Date(Number(value) * 1000);
    if (Number.isNaN(date.getTime())) return "";
    const pad = (part) => String(part).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  onMount(() => {
    let disposed = false;
    let source;
    let reconnectTimer;
    async function loadHistory() {
      try {
        const [nextRun, result] = await Promise.all([
          fetchRun(runId),
          fetchRunLogs(runId, { maxLines: 500, search, since, until }),
        ]);
        if (!disposed) {
          run = nextRun || run;
          if (!since) since = localRange(nextRun?.started_at || nextRun?.created_at);
          if (!until) until = localRange(nextRun?.ended_at || nextRun?.completed_at);
          lines = result.logs || [];
        }
      } catch (reason) {
        if (!disposed) error = reason?.message || String(reason);
      } finally {
        if (!disposed) loading = false;
      }
    }
    function connect() {
      if (run && !isRunning) {
        streamState = "stored logs";
        return;
      }
      source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/logs/stream`);
      streamState = "streaming";
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (!disposed) lines = [...lines, payload].slice(-2000);
        } catch {
          // Ignore malformed keepalive frames.
        }
      };
      source.onerror = () => {
        streamState = "reconnecting";
        source?.close();
        if (!disposed) reconnectTimer = window.setTimeout(connect, 1000);
      };
    }
    void loadHistory();
    connect();
    const timer = window.setInterval(loadHistory, 5000);
    return () => {
      disposed = true;
      source?.close();
      window.clearTimeout(reconnectTimer);
      window.clearInterval(timer);
    };
  });

  let filtered = $derived(
    lines.filter((entry) => !search || String(entry.line || entry).toLowerCase().includes(search.toLowerCase())),
  );

  function resetRange() {
    since = "";
    until = "";
  }
</script>

<div class="tab-panel">
  {#if isRunning}
    <div class="flex flex-wrap items-center gap-[8px] mb-[8px]">
      <input class="log-search flex-1 min-w-[180px]" bind:value={search} aria-label="Filter log lines" placeholder="filter substring…" />
      <span class="text-[10px] text-(--muted)">{streamState}</span>
    </div>
  {:else}
    <div class="flex flex-col gap-[12px] mb-[12px] p-[12px_14px] rounded-[8px] bg-(--color-c-gray-08,#161616) [border:1px_solid_var(--border,#2f2f2f)]">
      <div class="flex items-center gap-[12px]">
        <input
          class="flex-1 min-w-0 bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[6px_10px] text-[12px] [font-family:inherit] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
          type="search"
          bind:value={search}
          aria-label="Filter log lines"
          placeholder="filter substring…"
        />
        <span class="inline-flex items-center gap-[6px] text-[11px] text-(--muted) uppercase tracking-[0.04em] shrink-0">
          <span class="dot dot-dim"></span> stored logs
        </span>
      </div>
      <div class="flex flex-wrap items-center gap-[10px]">
        <span class="text-(--muted) text-[11px] uppercase tracking-[0.04em]">Time range</span>
        <input
          class="w-[160px] bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] [font-variant-numeric:tabular-nums] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
          bind:value={since}
          aria-label="Show logs since"
          placeholder="YYYY-MM-DD HH:MM"
        />
        <span class="text-(--muted-strong) text-[13px]">→</span>
        <input
          class="w-[160px] bg-(--color-c-gray-10,#1c1c1c) text-(--text) [border:1px_solid_var(--border,#3a3a3a)] rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] [font-variant-numeric:tabular-nums] focus:outline-none focus:[border-color:color-mix(in_srgb,var(--accent)_55%,transparent)]"
          bind:value={until}
          aria-label="Show logs until"
          placeholder="YYYY-MM-DD HH:MM"
        />
        <button class="log-button text-[11px] px-[10px] py-[4px]" onclick={resetRange} title="Reset to the run's time range">Reset</button>
      </div>
    </div>
  {/if}
  {#if error}
    <div class="detail-empty">Failed to load logs: {error}</div>
  {:else if loading && !lines.length}
    <div class="detail-empty">Loading logs…</div>
  {:else if !filtered.length}
    <div class="detail-empty">{search ? `No log lines matching "${search}".` : "No logs recorded for this run."}</div>
  {:else}
    <div class="bg-(--color-c-gray-08,#0e0e0e) rounded-[6px] p-[8px_12px] max-h-[420px] overflow-y-auto overflow-x-auto [font-family:ui-monospace,SFMono-Regular,Menlo,monospace] text-[12px] leading-[1.45] text-(--text)">
      {#each filtered as entry, index (entry.id ?? index)}
        <div class="flex gap-[10px] whitespace-pre">
          <span class="shrink-0 text-(--muted) text-[10px] min-w-[64px] overflow-hidden text-ellipsis">{entry.task_id || ""}</span>
          <span class="flex-1 whitespace-pre-wrap break-all">{entry.line || entry}</span>
        </div>
      {/each}
    </div>
    <div class="mt-[6px] text-[11px] text-(--muted) [font-variant-numeric:tabular-nums]">
      Showing {filtered.length} line{filtered.length === 1 ? "" : "s"}
    </div>
  {/if}
</div>
