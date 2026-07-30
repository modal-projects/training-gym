<script>
  import { Download } from "lucide-svelte";
  import FrameworkActivityTrace from "./FrameworkActivityTrace.svelte";

  let {
    stepTimes = null,
    substepTimes = null,
    rolloutStats = [],
    layout = "rows",
    downloadName = "step_substep_times.json",
  } = $props();

  const LABELS = {
    evaluate_rollouts: "Eval (before)",
    generate_rollouts: "Generate rollouts",
    offload_rollout: "Offload rollout",
    compute_log_probs: "Compute log probs",
    optimizer_step: "Optimizer step",
    checkpoint_save: "Checkpoint save",
    offload_train: "Offload train",
    weight_sync: "Weight sync",
    evaluate_rollouts_end: "Eval (after)",
    full_step: "Full step",
    wait_for_rollout: "Wait for rollout",
    train_models: "Train models",
    train_model: "Train model",
    forward_backward: "Forward/backward",
    training_cleanup: "Training cleanup",
    wait_for_next_rollout: "Wait for next rollout",
    custom_reward: "Custom reward",
    custom_reward_post_process: "Reward post-process",
    evaluate_rollouts_before: "Eval (before)",
    evaluate_rollouts_after: "Eval (after)",
  };

  const COLORS = {
    evaluate_rollouts: "var(--color-c-dataviz-primary-7, #648fe0)",
    generate_rollouts: "var(--color-c-dataviz-primary-1, #adeaab)",
    offload_rollout: "var(--color-c-dataviz-paired-6, #859400)",
    compute_log_probs: "var(--color-c-dataviz-paired-5, #e6b687)",
    optimizer_step: "var(--color-c-dataviz-paired-7, #8956fa)",
    weight_sync: "var(--color-c-dataviz-primary-4, #4aa19d)",
    checkpoint_save: "var(--color-c-dataviz-primary-3, #ffc1f7)",
    offload_train: "var(--color-c-dataviz-paired-3, #ca70ad)",
    evaluate_rollouts_end: "var(--color-c-dataviz-primary-7, #648fe0)",
    full_step: "var(--color-c-gray-40, #747474)",
    wait_for_rollout: "var(--color-c-dataviz-paired-4, #6cabc1)",
    train_models: "var(--color-c-dataviz-primary-7, #648fe0)",
    train_model: "var(--color-c-dataviz-primary-7, #648fe0)",
    forward_backward: "var(--color-c-dataviz-paired-4, #6cabc1)",
    training_cleanup: "var(--color-c-dataviz-paired-3, #ca70ad)",
    wait_for_next_rollout: "var(--color-c-dataviz-paired-4, #6cabc1)",
    custom_reward: "var(--color-c-dataviz-primary-4, #4aa19d)",
    custom_reward_post_process: "var(--color-c-dataviz-primary-5, #decb6c)",
    evaluate_rollouts_before: "var(--color-c-dataviz-primary-7, #648fe0)",
    evaluate_rollouts_after: "var(--color-c-dataviz-primary-7, #648fe0)",
  };

  const DESCRIPTIONS = {
    generate_rollouts: "Generates responses for this source rollout.",
    custom_reward: "Executes the run's custom reward function.",
    custom_reward_post_process: "Post-processes custom reward results.",
    train_models: "Driver wall time dispatching and waiting for model training.",
    train_model: "Training worker wall time for this rollout.",
    forward_backward:
      "Forward pass, loss computation, backward pass, and gradient preparation.",
    optimizer_step:
      "Host interval for optimizer.step(); CUDA work may complete asynchronously.",
    compute_log_probs: "Computes policy log probabilities.",
    checkpoint_save: "Persists a training checkpoint.",
    weight_sync: "Synchronizes updated model weights with rollout workers.",
    offload_rollout: "Offloads the rollout model.",
    offload_train: "Offloads the training model.",
    wait_for_rollout: "Driver waiting for the source rollout to finish.",
    wait_for_next_rollout: "Driver waiting for the next prefetched rollout.",
  };

  const PHASE_ORDER = Object.keys(LABELS);
  const ROLE_ORDER = ["rollout", "driver", "actor", "critic", "step"];
  const WAIT_PHASES = new Set(["wait_for_rollout", "wait_for_next_rollout"]);

  function parseSubstepName(name) {
    const match = name.match(/^(.*) \((.*)\)$/);
    return {
      phase: match?.[1] || name,
      role: match?.[2] || "step",
    };
  }

  function labelFor(phase) {
    return LABELS[phase] || phase.replace(/_/g, " ");
  }

  function colorFor(phase) {
    return COLORS[phase] || "var(--color-c-dataviz-primary-other, #6d6161)";
  }

  function descriptionFor(phase) {
    return DESCRIPTIONS[phase] || null;
  }

  function fmtSecs(value) {
    if (value == null) return "—";
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return "—";
    const trim = (number) => number.toFixed(3).replace(/\.?0+$/, "");
    if (seconds >= 60) {
      const minutes = Math.floor(seconds / 60);
      return `${minutes}m ${trim(seconds - minutes * 60)}s`;
    }
    return `${trim(seconds)}s`;
  }

  function downloadJson() {
    const payload = {
      step_times: stepTimes || {},
      substep_times: substepTimes || {},
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = downloadName;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  let steps = $derived.by(() => {
    const keys = Array.from(
      new Set([
        ...Object.keys(stepTimes || {}),
        ...Object.keys(substepTimes || {}),
      ]),
    );
    const parsed = keys.map((key) => {
      const step = (stepTimes || {})[key] || null;
      const substeps = Object.entries((substepTimes || {})[key] || {})
        .flatMap(([name, value]) => {
          const { phase, role } = parseSubstepName(name);
          const intervals = value?.intervals?.length
            ? value.intervals
            : [
                {
                  started_at_unix_s: value?.start,
                  duration_s: value?.duration_s,
                },
              ];
          return intervals.map((interval, index) => ({
            name: intervals.length > 1 ? `${name}:${index}` : name,
            phase,
            role,
            start: interval?.started_at_unix_s ?? null,
            duration: interval?.duration_s ?? null,
            timelineGroup: value?.timeline_group ?? null,
            activityKind: value?.activity_kind ?? null,
            displayName: value?.display_name ?? null,
            parentPhase: value?.parent_phase ?? null,
            activityRolloutId: value?.activity_rollout_id ?? null,
            activityRolloutKind: value?.activity_rollout_kind ?? null,
            sourceRolloutId: value?.source_rollout_id ?? null,
            trainingRolloutId: value?.training_rollout_id ?? null,
            clockUncertainty: value?.clock_uncertainty_s ?? null,
            executionSequence: value?.execution_sequence ?? null,
          }));
        })
        .sort((left, right) => (left.start ?? 0) - (right.start ?? 0));
      const phaseStarts = substeps
        .map((substep) => Number(substep.start))
        .filter(Number.isFinite);
      const phaseEnds = substeps
        .map(
          (substep) => Number(substep.start) + Number(substep.duration),
        )
        .filter(Number.isFinite);
      const timelineStart = phaseStarts.length
        ? Math.min(...phaseStarts)
        : step?.start ?? null;
      const timelineEnd = phaseEnds.length
        ? Math.max(...phaseEnds)
        : step?.end ?? null;
      const roleNames = Array.from(
        new Set(substeps.map((substep) => substep.role)),
      ).sort((left, right) => {
        const leftIndex = ROLE_ORDER.indexOf(left);
        const rightIndex = ROLE_ORDER.indexOf(right);
        return (
          (leftIndex < 0 ? ROLE_ORDER.length : leftIndex) -
          (rightIndex < 0 ? ROLE_ORDER.length : rightIndex)
        );
      });
      return {
        key,
        n: Number.isFinite(Number(key)) ? Number(key) : key,
        duration: step?.duration_s ?? null,
        timelineStart,
        timelineDuration:
          timelineStart != null && timelineEnd != null
            ? Math.max(timelineEnd - timelineStart, 0)
            : step?.duration_s ?? null,
        substeps,
        roles: roleNames.map((role) => ({
          role,
          substeps: substeps.filter((substep) => substep.role === role),
        })),
      };
    });
    return parsed.sort(
      (left, right) => (Number(left.key) || 0) - (Number(right.key) || 0),
    );
  });

  let legend = $derived.by(() => {
    const phases = new Set();
    for (const step of steps) {
      for (const substep of step.substeps) {
        if (
          layout === "timeline" &&
          substep.phase === "full_step" &&
          step.substeps.some(
            (candidate) =>
              candidate.role === substep.role &&
              candidate.phase !== "full_step",
          )
        ) {
          continue;
        }
        phases.add(substep.phase);
      }
    }
    return Array.from(phases).sort((left, right) => {
      const leftIndex = PHASE_ORDER.indexOf(left);
      const rightIndex = PHASE_ORDER.indexOf(right);
      return (
        (leftIndex < 0 ? PHASE_ORDER.length : leftIndex) -
        (rightIndex < 0 ? PHASE_ORDER.length : rightIndex)
      );
    });
  });

  function positionedSubsteps(step, role) {
    const duration = Number(step.timelineDuration);
    const start = Number(step.timelineStart);
    if (Number.isFinite(duration) && duration > 0 && Number.isFinite(start)) {
      return role.substeps.map((substep) => {
        const substepStart = Number(substep.start);
        const substepDuration = Number(substep.duration);
        if (!Number.isFinite(substepStart) || !Number.isFinite(substepDuration)) {
          return { ...substep, left: 0, width: 2 };
        }
        const left = Math.max(
          0,
          Math.min(100, ((substepStart - start) / duration) * 100),
        );
        return {
          ...substep,
          left,
          width: Math.max(
            0,
            Math.min(100 - left, (substepDuration / duration) * 100),
          ),
        };
      });
    }
    const total = role.substeps.reduce(
      (sum, substep) => sum + Math.max(Number(substep.duration) || 0, 0),
      0,
    );
    let elapsed = 0;
    return role.substeps.map((substep) => {
      const duration = Math.max(Number(substep.duration) || 0, 0);
      const positioned = {
        ...substep,
        left: total > 0 ? (elapsed / total) * 100 : 0,
        width: total > 0 ? (duration / total) * 100 : 100,
      };
      elapsed += duration;
      return positioned;
    });
  }
</script>

{#if steps.length}
  {#if layout === "timeline"}
    <FrameworkActivityTrace
      {steps}
      {stepTimes}
      {rolloutStats}
      {legend}
      phaseOrder={PHASE_ORDER}
      {labelFor}
      {colorFor}
      {descriptionFor}
      {downloadJson}
    />
  {:else}
    <div class="step-timings">
      <div class="header">
        <div class="legend">
          {#each legend as phase (phase)}
            <span class="legend-item">
              <span
                class="swatch"
                class:wait={WAIT_PHASES.has(phase)}
                style:background={colorFor(phase)}
              ></span>
              {labelFor(phase)}
            </span>
          {/each}
        </div>
        <button onclick={downloadJson} title="Download step + substep times as JSON">
          <Download size={13} />
          Download JSON
        </button>
      </div>
      {#each steps as step (step.key)}
        <div class="step-row">
          <div class="step-head">
            <span>Step {step.n}</span>
            <span class="duration">{fmtSecs(step.duration)}</span>
          </div>
          {#each step.roles as role (role.role)}
            <div class="role-row">
              <span class="role-label">
                {role.role === "step" ? "Timing" : role.role}
              </span>
              <div class="bar">
                {#each positionedSubsteps(step, role) as substep (substep.name)}
                  <div
                    class="segment"
                    class:wait={substep.activityKind
                      ? substep.activityKind === "wait"
                      : WAIT_PHASES.has(substep.phase)}
                    style:left={`${substep.left}%`}
                    style:width={`${substep.width}%`}
                    style:background={colorFor(substep.phase)}
                    title={`${substep.displayName || labelFor(substep.phase)} (${substep.role}) · ${fmtSecs(substep.duration)}`}
                  ></div>
                {/each}
              </div>
            </div>
          {/each}
        </div>
      {/each}
    </div>
  {/if}
{/if}

<style>
  .step-timings {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .header,
  .legend,
  .legend-item,
  button {
    display: flex;
    align-items: center;
  }

  .header {
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .legend {
    flex-wrap: wrap;
    gap: 6px 14px;
    color: var(--muted);
    font-size: 11px;
  }

  .legend-item {
    gap: 5px;
  }

  .swatch {
    width: 9px;
    height: 9px;
    flex-shrink: 0;
    border-radius: 2px;
  }

  .wait {
    background: repeating-linear-gradient(
      135deg,
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 65%, #181818),
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 65%, #181818) 3px,
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 20%, #181818) 3px,
      color-mix(in srgb, var(--color-c-dataviz-paired-4, #6cabc1) 20%, #181818) 6px
    ) !important;
  }

  button {
    flex-shrink: 0;
    gap: 5px;
    padding: 3px 8px;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    background: none;
    color: var(--muted);
    cursor: pointer;
    font-size: 11px;
  }

  .step-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .step-head {
    display: flex;
    justify-content: space-between;
    color: var(--text-bright);
    font-size: 12px;
    font-weight: 500;
  }

  .duration {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    font-weight: 400;
  }

  .role-row {
    display: grid;
    grid-template-columns: 48px minmax(0, 1fr);
    gap: 6px;
  }

  .role-label {
    overflow: hidden;
    color: var(--muted);
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.04em;
    line-height: 14px;
    text-align: right;
    text-overflow: ellipsis;
    text-transform: uppercase;
  }

  .bar {
    position: relative;
    height: 14px;
    overflow: hidden;
    border-radius: 3px;
    background: var(--color-c-gray-08, #1c1c1c);
  }

  .segment {
    position: absolute;
    top: 0;
    height: 100%;
    border-left: 1px solid color-mix(in srgb, #000 35%, transparent);
  }
</style>
