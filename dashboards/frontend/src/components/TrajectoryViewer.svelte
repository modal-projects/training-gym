<script>
  /**
   * Training Gym's framework-default trajectory viewer.
   *
   * This component intentionally knows nothing about a particular
   * environment or state schema. Frameworks can provide a richer viewer by
   * mounting a component with the same props.
   *
   * Props supplied by TrainingRunDetailPage:
   *   sample       - selected normalized sample object
   *   samples      - all samples in the selected rollout/prompt group
   *   trajectory   - sample.metadata.trajectory_messages[]
   *   rewardEvents - sample.reward_events[] (if emitted)
   *   rollout      - full expanded TrainingRolloutResult
   *   run          - current TrainingRun summary
   */
  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
  } = $props();

  function asText(value) {
    return typeof value === "string" ? value : value == null ? "" : String(value);
  }

  function sourceMessages() {
    if (Array.isArray(trajectory) && trajectory.length) return trajectory;
    const metadataMessages = sample?.metadata?.trajectory_messages;
    if (Array.isArray(metadataMessages) && metadataMessages.length) return metadataMessages;
    return [];
  }

  function parseStructured(content) {
    const text = asText(content).trim();
    if (!text) return null;
    const candidates = [text];
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced?.[1]) candidates.push(fenced[1].trim());
    const object = text.match(/\{[\s\S]*\}/);
    if (object?.[0]) candidates.push(object[0]);
    for (const candidate of candidates) {
      try {
        const value = JSON.parse(candidate);
        if (value && typeof value === "object") return value;
      } catch {
        // Natural-language messages are valid trajectory entries too.
      }
    }
    return null;
  }

  function frameLabel(message, index) {
    const role = asText(message?.role || "event").toUpperCase();
    return `${String(index + 1).padStart(2, "0")} · ${role}`;
  }

  function preview(message) {
    const text = asText(message?.content || message?.text).replace(/\s+/g, " ").trim();
    return text.length > 72 ? `${text.slice(0, 72)}…` : text || "structured event";
  }

  let frames = $derived(
    sourceMessages().map((message, index) => ({
      message,
      index,
      label: frameLabel(message, index),
      preview: preview(message),
      structured: parseStructured(message?.content || message?.text),
    })),
  );
  let activeIndex = $state(0);
  let activeFrame = $derived(frames[Math.min(activeIndex, Math.max(frames.length - 1, 0))] || null);
  let latestReward = $derived(Array.isArray(rewardEvents) ? rewardEvents.at(-1) : null);

  function selectFrame(index) {
    activeIndex = index;
  }
</script>

<section class="trajectory-viewer" data-trajectory-viewer="default">
  <header class="viewer-header">
    <div>
      <div class="viewer-kicker">TRAINING GYM · TRAJECTORY</div>
      <h3>Trajectory</h3>
      <p>{frames.length} events · {Array.isArray(samples) ? samples.length : 0} samples</p>
    </div>
    {#if run?.status}
      <span class="run-status">{run.status}</span>
    {/if}
  </header>

  {#if frames.length}
    <nav class="frame-strip" aria-label="Trajectory events">
      {#each frames as frame, index}
        <button
          class:active={index === activeIndex}
          class="frame-tab"
          title={frame.preview}
          onclick={() => selectFrame(index)}
        >
          <span class="frame-number">{String(index + 1).padStart(2, "0")}</span>
          <span class="frame-copy">
            <b>{asText(frame.message?.role || "event")}</b>
            <small>{frame.preview}</small>
          </span>
        </button>
      {/each}
    </nav>

    {#if activeFrame}
      <article class="event-card">
        <div class="event-heading">
          <span>{activeFrame.label}</span>
          {#if latestReward && activeFrame.index === frames.length - 1}
            <strong>{asText(latestReward.label || "reward")} · {Number(latestReward.reward || 0).toFixed(3)}</strong>
          {/if}
        </div>
        <pre class="event-content">{asText(activeFrame.message?.content || activeFrame.message?.text)}</pre>
        {#if activeFrame.structured}
          <details class="structured-state">
            <summary>Structured data</summary>
            <pre>{JSON.stringify(activeFrame.structured, null, 2)}</pre>
          </details>
        {/if}
      </article>
    {/if}
  {:else}
    <div class="empty-viewer">No trajectory events were recorded for this sample.</div>
  {/if}
</section>

<style>
  .trajectory-viewer { --viewer-panel: var(--panel, var(--color-c-gray-2, #1c1c1c)); --viewer-panel-alt: var(--panel-alt, var(--color-c-gray-5, #242424)); --viewer-border: var(--border, var(--color-c-gray-10, #2f2f2f)); --viewer-text: var(--text, var(--color-c-gray-80, #d1d1d1)); --viewer-muted: var(--muted, var(--color-c-gray-50, #8b8b8b)); margin: 12px 0 14px; overflow: hidden; border: 1px solid var(--viewer-border); border-radius: 8px; background: var(--viewer-panel); color: var(--viewer-text); box-shadow: 0 5px 18px rgb(0 0 0 / 18%); }
  .viewer-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; padding: 15px 16px 13px; border-bottom: 1px solid var(--viewer-border); }
  .viewer-kicker { color: var(--viewer-muted); font-size: 9px; font-weight: 700; letter-spacing: .14em; }
  h3 { margin: 4px 0 2px; font-size: 17px; font-weight: 650; letter-spacing: -.02em; }
  .viewer-header p { margin: 0; color: var(--viewer-muted); font-size: 10px; }
  .run-status { padding: 4px 7px; border: 1px solid var(--viewer-border); border-radius: 4px; color: var(--viewer-muted); font: 700 9px ui-monospace, monospace; text-transform: uppercase; }
  .frame-strip { display: flex; gap: 5px; overflow-x: auto; padding: 10px 12px; background: var(--viewer-panel-alt); scrollbar-width: thin; }
  .frame-tab { display: flex; min-width: 150px; align-items: center; gap: 8px; padding: 7px 9px; border: 1px solid var(--viewer-border); border-radius: 5px; background: var(--viewer-panel); color: var(--viewer-muted); text-align: left; cursor: pointer; }
  .frame-tab:hover { border-color: var(--border-strong, #4a4a4a); }
  .frame-tab.active { border-color: var(--accent, #7c9cff); background: color-mix(in srgb, var(--accent, #7c9cff) 12%, var(--viewer-panel)); color: var(--text-bright, #fff); box-shadow: inset 0 -2px var(--accent, #7c9cff); }
  .frame-number { color: var(--viewer-muted); font: 700 10px ui-monospace, monospace; }
  .frame-copy { min-width: 0; }
  .frame-copy b, .frame-copy small { display: block; white-space: nowrap; }
  .frame-copy b { overflow: hidden; font-size: 10px; font-weight: 650; text-overflow: ellipsis; text-transform: uppercase; }
  .frame-copy small { max-width: 125px; margin-top: 2px; overflow: hidden; color: var(--viewer-muted); font-size: 9px; text-overflow: ellipsis; }
  .event-card { margin: 0 12px 12px; border: 1px solid var(--viewer-border); border-radius: 6px; background: var(--viewer-panel-alt); }
  .event-heading { display: flex; justify-content: space-between; gap: 12px; padding: 9px 11px; border-bottom: 1px solid var(--viewer-border); color: var(--viewer-muted); font: 700 9px ui-monospace, monospace; text-transform: uppercase; }
  .event-heading strong { color: var(--green, #75d95c); font-weight: 700; }
  .event-content { max-height: 300px; overflow: auto; margin: 0; padding: 12px; color: var(--viewer-text); font: 11px/1.5 ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .structured-state { border-top: 1px solid var(--viewer-border); padding: 8px 11px 10px; }
  .structured-state summary { color: var(--viewer-muted); font-size: 10px; cursor: pointer; }
  .structured-state pre { max-height: 220px; overflow: auto; margin: 7px 0 0; color: var(--viewer-muted); font: 10px/1.45 ui-monospace, monospace; white-space: pre-wrap; }
  .empty-viewer { padding: 28px 22px; color: var(--viewer-muted); font-size: 11px; text-align: center; }
  @media (max-width: 600px) { .frame-tab { min-width: 130px; } .event-heading { display: grid; gap: 4px; } }
</style>
