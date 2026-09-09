<script>
  let { events = [], granularity = "transition" } = $props();

  let normalizedEvents = $derived(
    (Array.isArray(events) ? events : []).filter(
      (event) => event && Number.isFinite(Number(event.reward)),
    ),
  );
  let maxAbsReward = $derived(
    Math.max(
      0.001,
      ...normalizedEvents.map((event) => Math.abs(Number(event.reward) || 0)),
    ),
  );

  function format(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(3) : "—";
  }

  function barStyle(value) {
    const reward = Number(value) || 0;
    const width = (Math.abs(reward) / maxAbsReward) * 50;
    return reward >= 0
      ? `left:50%;width:${width}%`
      : `right:50%;width:${width}%`;
  }

  function componentText(event) {
    const components = event?.components;
    if (!components || typeof components !== "object") return "";
    return Object.entries(components)
      .map(([name, value]) => `${name}=${format(value)}`)
      .join(" · ");
  }
</script>

<div class="reward-timeline">
  <div class="reward-timeline-note">
    {granularity === "checkpoint" ? "Checkpoint-window credits" : "Transition deltas"}
    · {normalizedEvents.length} events · cumulative is the game total · scalar episode reward remains the training signal
  </div>
  <div class="reward-timeline-header">
    <span>turn</span>
    <span>{granularity === "checkpoint" ? "window reward" : "delta"}</span>
    <span>cumulative</span>
    <span>action-token span</span>
    <span>magnitude</span>
    <span>components</span>
  </div>
  <div class="reward-timeline-rows">
    {#each normalizedEvents as event, index (index)}
      {@const reward = Number(event.reward) || 0}
      <div class="reward-timeline-row" title={componentText(event)}>
        <span class="reward-timeline-turn">{event.turn ?? index + 1}</span>
        <span class:reward-positive={reward > 0} class:reward-negative={reward < 0} class="reward-timeline-delta">
          {reward > 0 ? "+" : ""}{format(reward)}
        </span>
        <span class="reward-timeline-cumulative">{format(event.cumulative_reward)}</span>
        <span class="reward-timeline-tokens">
          {event.token_start != null && event.token_end != null
            ? `${event.token_start}–${event.token_end}`
            : "—"}
        </span>
        <div class="reward-timeline-bar-track" aria-label={`reward ${format(reward)}`}>
          <div
            class:reward-bar-positive={reward >= 0}
            class:reward-bar-negative={reward < 0}
            class="reward-timeline-bar"
            style={barStyle(reward)}
          ></div>
          <div class="reward-timeline-zero"></div>
        </div>
        <span class="reward-timeline-components">{componentText(event) || event.label || "—"}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .reward-timeline {
    margin: 4px 0 10px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--color-c-gray-08, #1c1c1c);
    overflow: hidden;
  }

  .reward-timeline-note {
    padding: 7px 9px;
    color: var(--muted);
    font-size: 10px;
    border-bottom: 1px solid var(--border);
  }

  .reward-timeline-header,
  .reward-timeline-row {
    display: grid;
    grid-template-columns: 42px 68px 76px 92px minmax(90px, 0.7fr) minmax(150px, 1.3fr);
    gap: 7px;
    align-items: center;
    min-width: 470px;
  }

  .reward-timeline-header {
    padding: 5px 9px;
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
  }

  .reward-timeline-rows {
    max-height: 320px;
    overflow: auto;
  }

  .reward-timeline-row {
    padding: 4px 9px;
    color: var(--text);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }

  .reward-timeline-row:last-child {
    border-bottom: 0;
  }

  .reward-timeline-turn,
  .reward-timeline-tokens,
  .reward-timeline-components {
    color: var(--muted);
  }

  .reward-timeline-components {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .reward-timeline-delta,
  .reward-timeline-cumulative {
    text-align: right;
  }

  .reward-positive,
  .reward-bar-positive {
    color: var(--color-c-dataviz-primary-1, #adeaab);
  }

  .reward-negative,
  .reward-bar-negative {
    color: var(--color-c-dataviz-primary-2, #d9866b);
  }

  .reward-timeline-bar-track {
    position: relative;
    height: 12px;
    border-radius: 2px;
    background: #272727;
    overflow: hidden;
  }

  .reward-timeline-zero {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 1px;
    background: #777;
  }

  .reward-timeline-bar {
    position: absolute;
    top: 2px;
    height: 8px;
    border-radius: 2px;
  }

  .reward-bar-positive {
    background: var(--color-c-dataviz-primary-1, #adeaab);
  }

  .reward-bar-negative {
    background: var(--color-c-dataviz-primary-2, #d9866b);
  }
</style>
