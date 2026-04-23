<script>
  import {
    fetchHarborRun,
    fetchHarborRollouts,
    fetchHarborRollout,
  } from "./lib/api.js";
  import { fmtDuration } from "./lib/format.js";

  let { runId, onBack } = $props();

  let summary = $state(null);
  let rolloutIds = $state([]);
  let selectedRolloutId = $state(null);
  let rollout = $state(null);
  let selectedSampleIdx = $state(0);
  let selectedTurnIdx = $state(0);
  let loading = $state(true);
  let error = $state(null);

  async function load() {
    loading = true;
    error = null;
    try {
      const [sum, ids] = await Promise.all([
        fetchHarborRun(runId),
        fetchHarborRollouts(runId),
      ]);
      summary = sum;
      rolloutIds = ids;
      if (ids.length > 0) {
        await selectRollout(ids[ids.length - 1]);
      }
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  async function selectRollout(rid) {
    selectedRolloutId = rid;
    selectedSampleIdx = 0;
    selectedTurnIdx = 0;
    rollout = await fetchHarborRollout(runId, rid);
  }

  load();

  let samples = $derived(
    (rollout?.groups || []).flatMap((g) => g.samples || []),
  );
  let currentSample = $derived(samples[selectedSampleIdx] || null);
  let turns = $derived(currentSample?.trajectory_turns || []);
  let currentTurn = $derived(turns[selectedTurnIdx] || null);

  function fmtSec(s) {
    if (s == null || s === 0) return "–";
    return s < 1
      ? s.toFixed(2) + "s"
      : s < 60
        ? s.toFixed(1) + "s"
        : (s / 60).toFixed(1) + "m";
  }

  function roleCls(role) {
    if (role === "assistant") return "r-assistant";
    if (role === "user") return "r-user";
    if (role === "tool") return "r-tool";
    return "r-system";
  }

  function toolNames(turn) {
    if (!Array.isArray(turn?.tool_calls)) return [];
    return turn.tool_calls.map((tc) => {
      const fn = tc.function || tc;
      return fn.name || tc.name || "tool";
    });
  }

  function preview(turn) {
    const text = turn?.content || turn?.reasoning || "";
    return text.length > 80 ? text.slice(0, 80) + "…" : text;
  }

  function prettyJson(obj) {
    if (typeof obj === "string") {
      try {
        return JSON.stringify(JSON.parse(obj), null, 2);
      } catch {
        return obj;
      }
    }
    return JSON.stringify(obj, null, 2);
  }
</script>

<header class="topbar">
  <button class="btn" onclick={onBack}>&larr; Back</button>
  <h1>Harbor Run</h1>
  <span class="run-id">{runId}</span>
  {#if summary}
    <span class="status-badge status-{summary.status || 'unknown'}"
      >{summary.status || "unknown"}</span
    >
  {/if}
</header>

<div class="harbor-main">
  {#if loading}
    <div class="empty">Loading...</div>
  {:else if error}
    <div class="empty">Error: {error}</div>
  {:else}
    <!-- Summary bar -->
    {#if summary}
      <div class="summary-bar">
        <div class="stat-card">
          <div class="stat-val">{summary.rollout_count}</div>
          <div class="stat-lbl">Rollouts</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">{summary.sample_count}</div>
          <div class="stat-lbl">Samples</div>
        </div>
        <div class="stat-card">
          <div class="stat-val"
            >{(summary.reward_mean || 0).toFixed(3)}</div
          >
          <div class="stat-lbl">Reward (mean)</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">{summary.ok_count || 0}</div>
          <div class="stat-lbl">OK</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">{summary.failure_count || 0}</div>
          <div class="stat-lbl">Failed</div>
        </div>
      </div>
    {/if}

    <!-- Rollout selector -->
    {#if rolloutIds.length > 0}
      <div class="rollout-selector">
        <span class="section-label">Rollout</span>
        <div class="pill-group">
          {#each rolloutIds as rid (rid)}
            <button
              class="pill"
              class:active={rid === selectedRolloutId}
              onclick={() => selectRollout(rid)}
            >
              {rid}
            </button>
          {/each}
        </div>
      </div>
    {/if}

    {#if rollout && samples.length > 0}
      <!-- Sample tabs -->
      <div class="sample-tabs">
        {#each samples as sample, idx (idx)}
          <button
            class="sample-tab"
            class:active={idx === selectedSampleIdx}
            onclick={() => {
              selectedSampleIdx = idx;
              selectedTurnIdx = 0;
            }}
          >
            <span
              class="tab-dot"
              class:ok={sample.exit_status === "ok"}
              class:fail={sample.exit_status !== "ok"}
            ></span>
            <span>{sample.task_name || `sample ${idx + 1}`}</span>
            <span class="tab-reward"
              >{(sample.reward || 0).toFixed(3)}</span
            >
          </button>
        {/each}
      </div>

      <!-- Metrics chips -->
      {#if currentSample}
        {@const am = currentSample.agent_metrics || {}}
        <div class="metrics-bar">
          <span class="chip"
            ><span class="chip-lbl">reward</span>{(
              currentSample.reward || 0
            ).toFixed(3)}</span
          >
          <span class="chip"
            ><span class="chip-lbl">status</span>{currentSample.exit_status ||
              "unknown"}</span
          >
          {#if am.total_time}
            <span class="chip"
              ><span class="chip-lbl">total</span>{fmtSec(
                am.total_time,
              )}</span
            >
          {/if}
          {#if am.model_latency}
            <span class="chip"
              ><span class="chip-lbl">model</span>{fmtSec(
                am.model_latency,
              )}</span
            >
          {/if}
          {#if am.verifier_latency}
            <span class="chip"
              ><span class="chip-lbl">verifier</span>{fmtSec(
                am.verifier_latency,
              )}</span
            >
          {/if}
          {#if currentSample.task_name}
            <span class="chip"
              ><span class="chip-lbl">task</span>{currentSample.task_name}</span
            >
          {/if}
        </div>
      {/if}

      <!-- Trajectory viewer: split pane -->
      <div class="split-pane">
        <!-- Left: turn list -->
        <div class="pane-left">
          <div class="pane-header">
            Trajectory <span class="pane-count">{turns.length} turns</span>
          </div>
          <div class="turn-list">
            {#if turns.length === 0}
              <div class="empty-small">No structured turns available</div>
            {:else}
              {#each turns as turn, ti (ti)}
                <button
                  class="turn-item"
                  class:selected={ti === selectedTurnIdx}
                  onclick={() => (selectedTurnIdx = ti)}
                >
                  <span class="turn-role {roleCls(turn.role)}"
                    >{turn.role}</span
                  >
                  <span class="turn-preview">{preview(turn)}</span>
                  {#if turn.reasoning}
                    <span class="turn-badge think">think</span>
                  {/if}
                  {#each toolNames(turn) as tn}
                    <span class="turn-badge tool">{tn}</span>
                  {/each}
                  {#if turn.duration_sec}
                    <span class="turn-dur">{fmtSec(turn.duration_sec)}</span>
                  {/if}
                </button>
              {/each}
            {/if}
          </div>
        </div>

        <!-- Right: turn detail -->
        <div class="pane-right">
          <div class="pane-header">Detail</div>
          <div class="turn-detail">
            {#if !currentTurn}
              <div class="empty-small">Select a turn to view details</div>
            {:else}
              <!-- Turn meta -->
              <div class="detail-meta">
                <span class="turn-role {roleCls(currentTurn.role)}"
                  >{currentTurn.role}</span
                >
                {#if currentTurn.name}
                  <span class="detail-name">{currentTurn.name}</span>
                {/if}
                {#if currentTurn.duration_sec}
                  <span class="detail-dur"
                    >{fmtSec(currentTurn.duration_sec)}</span
                  >
                {/if}
                {#if currentTurn.token_usage}
                  {@const u = currentTurn.token_usage}
                  <span class="detail-tokens">
                    {#if u.reasoning}{u.reasoning} think{/if}
                    {#if u.reasoning && u.answer}{" · "}{/if}
                    {#if u.answer}{u.answer} out{/if}
                    {#if u.reasoning || u.answer}{" tokens"}{/if}
                  </span>
                {/if}
              </div>

              <!-- Thinking section -->
              {#if currentTurn.reasoning}
                <div class="detail-section thinking">
                  <div class="section-hdr">Thinking</div>
                  <pre>{currentTurn.reasoning}</pre>
                </div>
              {/if}

              <!-- Content section -->
              {#if currentTurn.content}
                <div
                  class="detail-section"
                  class:response={!!currentTurn.reasoning}
                >
                  <div class="section-hdr">
                    {currentTurn.reasoning ? "Response" : "Content"}
                  </div>
                  <pre>{currentTurn.content}</pre>
                </div>
              {/if}

              <!-- Tool calls -->
              {#if currentTurn.tool_calls?.length}
                <div class="detail-section">
                  <div class="section-hdr">
                    Tool Calls ({currentTurn.tool_calls.length})
                  </div>
                  {#each currentTurn.tool_calls as tc}
                    {@const fn = tc.function || tc}
                    <div class="tool-call">
                      <div class="tool-name">
                        {fn.name || tc.name || "tool"}
                      </div>
                      <pre class="tool-args"
                        >{prettyJson(
                          fn.arguments || fn.input || tc.arguments || "",
                        )}</pre>
                    </div>
                  {/each}
                </div>
              {/if}
            {/if}
          </div>
        </div>
      </div>

      <!-- Eval report -->
      {#if currentSample?.eval_report && Object.keys(currentSample.eval_report).length}
        <details class="collapsible">
          <summary>Verifier Report</summary>
          <pre>{prettyJson(currentSample.eval_report)}</pre>
        </details>
      {/if}

      <!-- Raw prompt -->
      {#if currentSample?.prompt}
        <details class="collapsible">
          <summary>Prompt</summary>
          <pre>{currentSample.prompt}</pre>
        </details>
      {/if}

      <!-- Raw metadata -->
      {#if currentSample?.metadata}
        <details class="collapsible">
          <summary>Raw Metadata</summary>
          <pre>{prettyJson(currentSample.metadata)}</pre>
        </details>
      {/if}
    {:else if rollout}
      <div class="empty">No samples in this rollout.</div>
    {:else if rolloutIds.length === 0}
      <div class="empty">
        No rollout data recorded yet. Observability data appears after the
        first training rollout completes.
      </div>
    {/if}
  {/if}
</div>

<style>
  .topbar {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 1.25rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .topbar h1 {
    font-size: 1.25rem;
    font-weight: 700;
  }
  .run-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85em;
    color: var(--accent);
  }
  .status-badge {
    padding: 0.15rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.8em;
    font-weight: 500;
  }
  .status-running {
    background: rgba(74, 222, 128, 0.12);
    color: var(--green);
  }
  .status-completed {
    background: rgba(125, 211, 252, 0.12);
    color: var(--accent);
  }
  .status-failed {
    background: rgba(248, 113, 113, 0.12);
    color: var(--red);
  }
  .btn {
    background: #1f2937;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    cursor: pointer;
    font: inherit;
    font-size: 0.85em;
  }
  .btn:hover {
    background: #374151;
  }
  .harbor-main {
    padding: 1.5rem 2rem 3rem;
    max-width: 1400px;
  }
  .empty {
    color: var(--muted);
    padding: 3rem;
    text-align: center;
  }
  .empty-small {
    color: var(--muted);
    padding: 1rem;
    font-size: 0.9em;
    font-style: italic;
  }

  /* Summary */
  .summary-bar {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
  }
  .stat-card {
    flex: 1;
    min-width: 100px;
    text-align: center;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 0.5rem;
  }
  .stat-val {
    font-size: 1.25rem;
    font-weight: 700;
  }
  .stat-lbl {
    font-size: 0.7em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-top: 0.2rem;
  }

  /* Rollout selector */
  .rollout-selector {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }
  .section-label {
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 600;
  }
  .pill-group {
    display: flex;
    gap: 0.3rem;
    flex-wrap: wrap;
  }
  .pill {
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font: inherit;
    font-size: 0.8em;
    cursor: pointer;
  }
  .pill:hover {
    border-color: var(--accent);
    color: var(--text);
  }
  .pill.active {
    background: var(--accent-dim);
    border-color: var(--accent);
    color: var(--accent);
  }

  /* Sample tabs */
  .sample-tabs {
    display: flex;
    gap: 0.4rem;
    overflow-x: auto;
    padding-bottom: 0.5rem;
    margin-bottom: 0.75rem;
  }
  .sample-tab {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 0.85em;
    cursor: pointer;
    white-space: nowrap;
  }
  .sample-tab:hover {
    background: #1f2937;
  }
  .sample-tab.active {
    background: var(--accent-dim);
    border-color: var(--accent);
  }
  .tab-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .tab-dot.ok {
    background: var(--green);
  }
  .tab-dot.fail {
    background: var(--red);
  }
  .tab-reward {
    font-family: ui-monospace, monospace;
    font-size: 0.8em;
    color: var(--muted);
  }

  /* Metrics bar */
  .metrics-bar {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-family: ui-monospace, monospace;
    font-size: 0.8em;
    background: #1f2937;
    border: 1px solid var(--border);
  }
  .chip-lbl {
    color: var(--muted);
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* Split pane */
  .split-pane {
    display: flex;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    height: 520px;
    margin-bottom: 1rem;
    background: var(--panel);
  }
  .pane-left {
    width: 320px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
  }
  .pane-right {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  .pane-header {
    padding: 0.6rem 0.85rem;
    font-weight: 600;
    font-size: 0.85em;
    border-bottom: 1px solid var(--border);
    background: #1a1f2a;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .pane-count {
    font-weight: 400;
    font-size: 0.85em;
    color: var(--muted);
  }

  /* Turn list */
  .turn-list {
    overflow-y: auto;
    flex: 1;
  }
  .turn-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    cursor: pointer;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    border-left: 3px solid transparent;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 0.85em;
    text-align: left;
    width: 100%;
  }
  .turn-item:hover {
    background: rgba(125, 211, 252, 0.04);
  }
  .turn-item.selected {
    background: rgba(125, 211, 252, 0.08);
    border-left-color: var(--accent);
  }
  .turn-role {
    font-weight: 600;
    font-size: 0.7em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.1rem 0.35rem;
    border-radius: 3px;
    flex-shrink: 0;
  }
  .r-assistant {
    background: rgba(74, 222, 128, 0.12);
    color: var(--green);
  }
  .r-user {
    background: rgba(125, 211, 252, 0.12);
    color: var(--accent);
  }
  .r-tool {
    background: rgba(251, 191, 36, 0.12);
    color: var(--yellow);
  }
  .r-system {
    background: rgba(156, 163, 175, 0.12);
    color: var(--muted);
  }
  .turn-preview {
    flex: 1;
    min-width: 0;
    font-size: 0.85em;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .turn-badge {
    font-family: ui-monospace, monospace;
    font-size: 0.7em;
    padding: 0.05rem 0.3rem;
    border-radius: 3px;
    flex-shrink: 0;
  }
  .turn-badge.think {
    color: #c084fc;
    background: rgba(192, 132, 252, 0.1);
  }
  .turn-badge.tool {
    color: var(--yellow);
    background: rgba(251, 191, 36, 0.1);
  }
  .turn-dur {
    font-family: ui-monospace, monospace;
    font-size: 0.7em;
    color: var(--muted);
    flex-shrink: 0;
    margin-left: auto;
  }

  /* Turn detail */
  .turn-detail {
    overflow-y: auto;
    flex: 1;
  }
  .detail-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 0.85rem;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .detail-name {
    font-family: ui-monospace, monospace;
    font-size: 0.85em;
    color: var(--muted);
  }
  .detail-dur {
    font-family: ui-monospace, monospace;
    font-size: 0.85em;
    color: var(--accent);
    font-weight: 600;
    margin-left: auto;
  }
  .detail-tokens {
    font-family: ui-monospace, monospace;
    font-size: 0.75em;
    color: var(--muted);
  }
  .detail-section {
    border-bottom: 1px solid var(--border);
  }
  .section-hdr {
    padding: 0.5rem 0.85rem;
    font-size: 0.7em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    background: rgba(255, 255, 255, 0.02);
  }
  .detail-section pre {
    margin: 0;
    padding: 0.75rem 0.85rem;
    font-size: 0.8em;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
    background: transparent;
    color: var(--text);
    border: none;
  }
  .detail-section.thinking {
    border-left: 3px solid rgba(192, 132, 252, 0.4);
  }
  .detail-section.thinking .section-hdr {
    color: #c084fc;
    background: rgba(192, 132, 252, 0.05);
  }
  .detail-section.thinking pre {
    color: #e9d5ff;
    background: rgba(15, 15, 20, 0.4);
  }
  .detail-section.response {
    border-left: 3px solid rgba(74, 222, 128, 0.4);
  }
  .detail-section.response .section-hdr {
    color: var(--green);
    background: rgba(74, 222, 128, 0.05);
  }
  .tool-call {
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }
  .tool-call:last-child {
    border-bottom: none;
  }
  .tool-name {
    padding: 0.4rem 0.85rem;
    font-family: ui-monospace, monospace;
    font-size: 0.85em;
    font-weight: 600;
    color: var(--yellow);
    background: rgba(251, 191, 36, 0.04);
    border-bottom: 1px solid rgba(251, 191, 36, 0.1);
  }
  .tool-args {
    margin: 0;
    padding: 0.5rem 0.85rem;
    font-size: 0.8em;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    background: transparent;
    color: var(--text);
    border: none;
  }

  /* Collapsible details */
  .collapsible {
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.75rem;
    overflow: hidden;
    background: var(--panel);
  }
  .collapsible summary {
    cursor: pointer;
    padding: 0.6rem 0.85rem;
    font-weight: 600;
    font-size: 0.9em;
  }
  .collapsible pre {
    margin: 0;
    padding: 0.75rem 0.85rem;
    font-size: 0.8em;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    border-top: 1px solid var(--border);
    max-height: 400px;
    overflow-y: auto;
  }
</style>
