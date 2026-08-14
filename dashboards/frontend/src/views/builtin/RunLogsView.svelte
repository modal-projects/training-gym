<script>
  import { onMount } from "svelte";
  import { fetchRun, fetchRunLogs } from "../../lib/api.js";

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
    <div class="rounded-[6px] [border:1px_solid_var(--border,#2f2f2f)] bg-(--color-c-gray-08,#1c1c1c) p-[8px] mb-[8px]">
      <div class="flex flex-wrap items-center gap-[8px]">
        <input class="log-search flex-1 min-w-[180px]" bind:value={search} aria-label="Filter log lines" placeholder="filter substring…" />
        <span class="dot dot-dim"></span><span class="text-[10px] text-(--muted) uppercase">stored logs</span>
      </div>
      <div class="flex flex-wrap items-center gap-[6px] mt-[8px]">
        <span class="text-[10px] text-(--muted) uppercase">time range</span>
        <input class="log-range-input" bind:value={since} aria-label="Show logs since" placeholder="YYYY-MM-DD HH:MM" />
        <span class="text-(--muted)">→</span>
        <input class="log-range-input" bind:value={until} aria-label="Show logs until" placeholder="YYYY-MM-DD HH:MM" />
        <button class="log-button text-[11px] px-[10px] py-[4px]" onclick={resetRange}>Reset</button>
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
          <span class="text-(--muted)">{entry.ts ? new Date(Number(entry.ts) * 1000).toLocaleTimeString() : ""}</span>
          <span>{entry.line || entry}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>
