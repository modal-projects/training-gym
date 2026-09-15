<script>
  import { ChevronLeft, ChevronRight, Download, PanelRightClose } from "lucide-svelte";
  import Drawer from "./Drawer.svelte";
  import RunTimeline from "./RunTimeline.svelte";
  import InferenceStats from "./InferenceStats.svelte";
  import SampleTimeline from "./SampleTimeline.svelte";
  import ConversationView from "./ConversationView.svelte";
  import { groupByRollout, rolloutIndex, rolloutScores } from "../lib/rolloutGrouping.js";

  let {
    runId,
    rolloutId,
    // Row from the rollouts table (mean, episode_count, ...); may be null.
    summary = null,
    // Full rollout detail from `fetchRollout`; null while loading.
    rollout = null,
    loading = false,
    // Substep timing for this rollout only, or null.
    timings = null,
    timelineAsync = false,
    timelineRunOrigin = null,
    formatMean,
    onclose = () => {},
  } = $props();

  const BUCKET_COUNT = 12;
  let activeBucket = $state(null); // histogram bucket index, or null
  let activeSamplePos = $state(0); // position within the active bucket's list
  let detailsEl = $state(null);

  // Reset the drill-in whenever a different rollout is shown.
  $effect(() => {
    rolloutId;
    activeBucket = null;
    activeSamplePos = 0;
  });

  let sampleDist = $derived.by(() => {
    const samples = rollout?.samples || [];
    const rollouts = groupByRollout(samples);
    if (!rollouts.length) return null;
    const scores = rolloutScores(samples, rollouts);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of scores) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const count = lo === hi ? 1 : BUCKET_COUNT;
    const span = hi - lo || 1;
    const buckets = Array.from({ length: count }, () => []);
    rollouts.forEach((positions, r) => {
      let b = count === 1 ? 0 : Math.floor(((scores[r] - lo) / span) * count);
      b = Math.max(0, Math.min(count - 1, b));
      buckets[b].push({ positions, score: scores[r] });
    });
    const maxCount = Math.max(...buckets.map((b) => b.length), 1);
    return {
      lo,
      hi,
      count,
      span,
      buckets,
      maxCount,
      total: rollouts.length,
      sampleCount: samples.length,
    };
  });

  // Open the first populated bucket once the detail lands.
  $effect(() => {
    const d = sampleDist;
    if (!d || activeBucket != null) return;
    const first = d.buckets.findIndex((b) => b.length > 0);
    if (first >= 0) openBucket(first);
  });

  let distSummary = $derived(!sampleDist ? "" : plural(sampleDist.total, "rollout"));

  function plural(n, unit) {
    return `${n} ${unit}${n === 1 ? "" : "s"}`;
  }

  function bucketLabel(bucket, b) {
    return `${plural(bucket.length, "rollout")} · reward ${bucketRange(b)}`;
  }

  function bucketRange(b) {
    const d = sampleDist;
    if (!d) return "";
    if (d.count === 1) return formatMean(d.lo);
    const step = d.span / d.count;
    return `${formatMean(d.lo + b * step)}–${formatMean(d.lo + (b + 1) * step)}`;
  }

  function openBucket(b) {
    const d = sampleDist;
    if (!d || !d.buckets[b]?.length) return;
    activeBucket = b;
    activeSamplePos = 0;
    scrollDetailsTop();
  }

  function stepSample(delta) {
    const d = sampleDist;
    if (!d || activeBucket == null) return;
    const list = d.buckets[activeBucket] || [];
    if (!list.length) return;
    activeSamplePos = Math.max(0, Math.min(list.length - 1, activeSamplePos + delta));
    scrollDetailsTop();
  }

  function scrollDetailsTop() {
    if (detailsEl) detailsEl.scrollTop = 0;
  }

  // A prompt group shares one screenshot: bytes on the first sample as `image`, the
  // rest carry only `image_ref`.
  let rolloutImages = $derived.by(() => {
    const byRef = {};
    for (const s of rollout?.samples ?? []) {
      const meta = s?.metadata;
      if (meta?.image_ref && meta.image) byRef[meta.image_ref] = meta.image;
    }
    return byRef;
  });

  function sampleImage(sample) {
    const meta = sample?.metadata;
    if (!meta) return null;
    return meta.image ?? (meta.image_ref ? rolloutImages[meta.image_ref] : null) ?? null;
  }

  // The sample currently shown in the viewer (or null when no bucket is open).
  let activeSample = $derived.by(() => {
    const d = sampleDist;
    if (!d || activeBucket == null) return null;
    const list = d.buckets[activeBucket] || [];
    const entry = list[activeSamplePos];
    if (!entry) return null;
    const sample = rollout.samples[entry.positions[0]];
    return {
      sample,
      samples: entry.positions.map((p) => rollout.samples[p]),
      score: entry.score,
      image: sampleImage(sample),
      pos: activeSamplePos,
      count: list.length,
    };
  });

  let diagnostics = $derived.by(() => {
    const m = rollout?.metrics;
    if (!m || !Object.keys(m).length) return null;
    const remoteErr = Number(m["agent/exit_status/remoteerror_sample_count"]) || 0;
    const responseMissing = Number(m["agent/response_missing_sample_count"]) || 0;
    const infraInvalid = Number(m["agent/invalid_infra_sample_count"]) || 0;
    const limitsExceeded = Number(m["agent/limits_exceeded_sample_count"]) || 0;
    const totalSamples = Number(m["agent/valid_sample_count"]) || sampleDist?.sampleCount || 0;
    if (!(remoteErr > 0 || responseMissing > 0 || infraInvalid > 0)) return null;
    return { remoteErr, responseMissing, infraInvalid, limitsExceeded, totalSamples };
  });

  function onKeydown(e) {
    if (activeBucket == null) return;
    const tag = (e.target?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") return;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepSample(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      stepSample(1);
    }
  }

  // `imageHandling`: "ignore" | "refs_only" | "resolve".
  function sampleToPayload(s, imageHandling = "ignore") {
    let metadata = s.metadata || null;
    if (imageHandling === "resolve" && metadata?.image_ref && !metadata.image) {
      // Only add bytes if the lookup resolved — the carrier sample may not be loaded.
      const resolved = sampleImage(s);
      if (resolved) metadata = { ...metadata, image: resolved };
    } else if (imageHandling === "refs_only" && metadata?.image && metadata.image_ref) {
      // Bytes travel once in the payload's `images` map; keep only the ref here.
      const { image, ...rest } = metadata;
      metadata = rest;
    }
    return {
      score: s.score,
      rollout_index: rolloutIndex(s),
      sample_index: s.sample_index ?? null,
      group_index: s.group_index ?? null,
      prompt: s.prompt || null,
      response: s.response || null,
      thinking: s.thinking || null,
      raw_response: s.raw_response || null,
      raw_prompt: s.raw_prompt || null,
      trace: s.trace || null,
      metadata,
    };
  }

  function downloadJson(payload, name) {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadSampleTrajectory() {
    if (!activeSample) return;
    const turns = activeSample.samples;
    const payload =
      turns.length === 1
        ? sampleToPayload(turns[0], "resolve")
        : {
            mean: activeSample.score,
            turns: turns.length,
            samples: turns.map((s) => sampleToPayload(s, "resolve")),
          };
    const r = rolloutId ?? 0;
    downloadJson(
      payload,
      `trajectory_r${r}_rollout${rolloutIndex(activeSample.sample) ?? activeSample.pos}.json`,
    );
  }

  function downloadAllTrajectories() {
    if (!rollout?.samples?.length) return;
    const r = rolloutId ?? 0;
    const samples = rollout.samples;
    const groups = groupByRollout(samples);
    const scores = rolloutScores(samples, groups);
    downloadJson(
      {
        training_run_id: runId,
        rollout_id: r,
        total: samples.length,
        rollouts: groups.length,
        n_samples_per_prompt: rollout.n_samples_per_prompt ?? null,
        mean: scores.reduce((a, v) => a + v, 0) / scores.length,
        // Shared images, keyed by the `metadata.image_ref` each sample carries.
        images: rolloutImages,
        samples: samples.map((s) => sampleToPayload(s, "refs_only")),
      },
      `rollout_${runId}_r${r}.json`,
    );
  }

  const KNOWN_METADATA_KEYS = [
    "inference",
    "_metadata_type",
    "audio",
    "image",
    "image_ref",
    "trajectory_messages",
    "eval_report",
    "reference",
    "metrics",
    "exit_status",
    "eval_detail",
    "response_length",
    "prompt_length",
    "rollout_id",
    "rollout_idx",
  ];
</script>

<svelte:window onkeydown={onKeydown} />

<Drawer open={rolloutId != null} {onclose} width="min(760px, 100vw)">
  <div class="h-full flex flex-col min-h-0" aria-label={`Rollout ${rolloutId}`}>
    <!-- Header -->
    <div class="p-[20px_24px_12px] flex justify-between [align-items:flex-start] gap-[12px] shrink-0">
      <div class="min-w-0">
        <span class="text-(--muted) text-[12px] leading-[16px] uppercase tracking-[0.04em]">Rollout</span>
        <div class="flex items-center gap-[10px] mt-[2px]">
          <h2 class="text-(--text-bright) [font-family:var(--font-mono)] text-[20px] font-normal leading-[28px]">#{rolloutId}</h2>
          {#if summary}
            <span class="text-[12px] text-(--muted) [font-variant-numeric:tabular-nums]">
              mean {formatMean(summary.mean)}{#if summary.episode_count != null}{` · ${plural(summary.episode_count, "rollout")}`}{/if}
            </span>
          {/if}
        </div>
      </div>
      <div class="flex items-center gap-[10px] shrink-0">
        {#if sampleDist}
          <button
            type="button"
            class="inline-flex items-center gap-[5px] [background:none] [border:1px_solid_var(--border,#2f2f2f)] rounded-[4px] text-(--muted) text-[11px] p-[3px_8px] cursor-pointer hover:text-(--text) hover:border-(--border-strong,#4a4a4a)"
            onclick={downloadAllTrajectories}
            title="Download all samples as JSON"
          >
            <Download size={13} />
            Download all ({sampleDist.sampleCount})
          </button>
        {/if}
        <button
          class="[border:0] [background:transparent] text-(--muted) cursor-pointer inline-flex items-center p-0 hover:text-(--text-bright)"
          onclick={onclose}
          aria-label="Close drawer"
        >
          <PanelRightClose size={20} />
        </button>
      </div>
    </div>

    {#if loading}
      <div class="detail-empty px-[24px]">Loading rollout…</div>
    {:else if !rollout || !sampleDist}
      <div class="detail-empty px-[24px]">No samples recorded.</div>
    {:else}
      <!-- Top: timing + diagnostics + histogram -->
      <div class="p-[0_24px] shrink-0 max-h-[55vh] overflow-y-auto overscroll-contain">
        {#if timings}
          <div class="mb-[12px]">
            <RunTimeline
              timings={{ [rolloutId]: timings }}
              asyncOverride={timelineAsync}
              runOrigin={timelineRunOrigin}
              showOpenRollout={false}
              timelineKey={`${runId}:${rolloutId}`}
              downloadName={`substep_timing_${runId}_rollout_${rolloutId}.json`}
              rolloutIds={[rolloutId]}
            />
          </div>
        {/if}
        {#if diagnostics}
          {@const d = diagnostics}
          <div class="rollout-diagnostics" class:diag-critical={d.remoteErr >= d.totalSamples}>
            <div class="diag-title">
              {#if d.remoteErr >= d.totalSamples}
                All {d.totalSamples} samples hit infrastructure errors
              {:else}
                {d.remoteErr + d.infraInvalid} / {d.totalSamples} samples hit infrastructure errors
              {/if}
            </div>
            <div class="flex flex-wrap gap-[6px] mb-[4px]">
              {#if d.remoteErr}
                <span class="diag-tag">RemoteError: {d.remoteErr}</span>
              {/if}
              {#if d.responseMissing}
                <span class="diag-tag">Response missing: {d.responseMissing}</span>
              {/if}
              {#if d.infraInvalid}
                <span class="diag-tag">Infra invalid: {d.infraInvalid}</span>
              {/if}
              {#if d.limitsExceeded}
                <span class="diag-tag">Limits exceeded: {d.limitsExceeded}</span>
              {/if}
            </div>
            {#if d.remoteErr >= d.totalSamples}
              <div class="text-[11px] text-(--muted,#a3a3a3) mt-[6px]">
                Check the Modal app logs for sandbox or image build errors.
              </div>
            {/if}
          </div>
        {/if}

        <div class="chart-scroll">
          <div
            class="flex items-end gap-[2px] h-[120px] pt-[14px] min-w-[280px] [border-bottom:1px_solid_var(--border,#2f2f2f)]"
            role="group"
            aria-label="Reward distribution"
          >
            {#each sampleDist.buckets as bucket, b (b)}
              <button
                type="button"
                class="dist-bar"
                class:detail-active={activeBucket === b}
                class:is-empty={!bucket.length}
                style:height={`${(bucket.length / sampleDist.maxCount) * 100}%`}
                disabled={!bucket.length}
                title={bucketLabel(bucket, b)}
                onclick={() => openBucket(b)}
              >
                <span class="absolute top-[-14px] left-0 right-0 text-center text-[10px] text-(--muted) [font-variant-numeric:tabular-nums]">{bucket.length || ""}</span>
              </button>
            {/each}
          </div>
          <div class="dist-axis">
            <span>{formatMean(sampleDist.lo)}</span>
            <span class="dist-axis-label">reward · {distSummary}</span>
            <span>{formatMean(sampleDist.hi)}</span>
          </div>
        </div>

      </div>

      <!-- Toolbar -->
      <div class="sample-toolbar shrink-0">
        {#if activeSample}
          <div class="sample-viewer-nav">
            <button
              class="sample-nav-btn"
              onclick={() => stepSample(-1)}
              disabled={activeSample.pos === 0}
              aria-label="Previous sample"
            >
              <ChevronLeft size={14} />
            </button>
            <span class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
              Sample {activeSample.pos + 1} of {activeSample.count}
            </span>
            <button
              class="sample-nav-btn"
              onclick={() => stepSample(1)}
              disabled={activeSample.pos === activeSample.count - 1}
              aria-label="Next sample"
            >
              <ChevronRight size={14} />
            </button>
            <span class="sample-viewer-hint">← / → to navigate</span>
          </div>
          <div class="sample-viewer-actions">
            <span class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
              reward {formatMean(activeSample.score)}{activeSample.samples.length > 1
                ? ` · first of ${activeSample.samples.length} turns`
                : ""}
            </span>
            <button
              class="sample-nav-btn"
              onclick={downloadSampleTrajectory}
              aria-label="Download trajectory JSON"
              title="Download trajectory"
            >
              <Download size={14} />
            </button>
          </div>
        {:else}
          <span class="text-[12px] text-(--muted)">Click a bar to inspect its samples.</span>
        {/if}
      </div>

      <!-- Sample details: independent scroll container -->
      <div class="flex-1 min-h-0 overflow-y-auto overscroll-contain p-[4px_24px_24px]" bind:this={detailsEl}>
        {#if activeSample}
          {#if activeSample.sample.metadata?.inference}
            <div class="rollout-sample-label">inference</div>
            <InferenceStats inference={activeSample.sample.metadata.inference} />
          {/if}
          {#if activeSample.sample.metadata?._metadata_type === "audio" || activeSample.sample.metadata?.audio}
            <div class="rollout-sample-label">audio</div>
            <audio
              class="block w-full max-w-[400px] m-[4px_0_8px] rounded-[4px]"
              controls
              preload="none"
              src={activeSample.sample.metadata.audio}
            ></audio>
          {/if}
          {#if activeSample.image}
            <div class="rollout-sample-label">image</div>
            <img
              class="block w-full max-w-[400px] h-auto m-[4px_0_8px] rounded-[4px] [border:1px_solid_var(--border)]"
              src={activeSample.image}
              alt="rollout input"
              loading="lazy"
            />
          {/if}
          {#if activeSample.sample.prompt}
            <div class="rollout-sample-label">prompt</div>
            <pre class="rollout-sample-text">{activeSample.sample.prompt}</pre>
          {/if}
          <div class="rollout-sample-label">conversation</div>
          <ConversationView
            messages={activeSample.sample.metadata?.trajectory_messages}
            response={activeSample.sample.response || ""}
            thinking={activeSample.sample.thinking || ""}
            evalReport={activeSample.sample.metadata?.eval_report}
            clamp={false}
          />
          {#if activeSample.sample.metadata?.reference}
            <div class="rollout-sample-label">reference</div>
            <pre class="rollout-sample-text">{activeSample.sample.metadata.reference}</pre>
          {/if}
          {#each Object.entries(activeSample.sample.metadata?.metrics ?? {}) as [name, value]}
            <div class="rollout-sample-label">{name}</div>
            <span class="rollout-sample-metric">
              {typeof value === "number" ? value.toFixed(3) : value}
            </span>
          {/each}
          {#if activeSample.sample.metadata?.exit_status}
            <div class="rollout-sample-label">exit status</div>
            <span class="rollout-sample-metric p-[2px_8px] rounded-[3px] text-[11px]! font-medium" class:exit-ok={activeSample.sample.metadata.exit_status === "ok"} class:exit-err={activeSample.sample.metadata.exit_status !== "ok"}>
              {activeSample.sample.metadata.exit_status}
            </span>
          {/if}
          {#if activeSample.sample.metadata?.eval_detail}
            <div class="rollout-sample-label">failure reason</div>
            <pre class="rollout-sample-text">{activeSample.sample.metadata.eval_detail}</pre>
          {/if}
          <!-- Catch-all: any other tag a custom reward/rollout function set on
               sample.metadata (e.g. sample.metadata["guessing"] = {...}) that
               isn't one of the known keys rendered explicitly above. -->
          {#each Object.entries(activeSample.sample.metadata ?? {}).filter(
            ([key]) => !KNOWN_METADATA_KEYS.includes(key),
          ) as [name, value] (name)}
            <div class="rollout-sample-label">{name}</div>
            {#if value !== null && typeof value === "object"}
              <pre class="rollout-sample-text">{JSON.stringify(value, null, 2)}</pre>
            {:else}
              <span class="rollout-sample-metric">{String(value)}</span>
            {/if}
          {/each}
          {#if activeSample.sample.trace?.length}
            <div class="rollout-sample-label">trajectory timeline</div>
            <div class="chart-scroll">
              <SampleTimeline trace={activeSample.sample.trace} />
            </div>
          {/if}
        {/if}
      </div>
    {/if}
  </div>
</Drawer>
