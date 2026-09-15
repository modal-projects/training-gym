<script>
  /**
   * Training Gym's built-in trajectory viewer.
   *
   * This is the slot that `training-gym setup --trajectory-viewer` replaces
   * at build time, and the fallback when a run has no run-scoped viewer. It
   * renders the conversation exactly as the rollout page always has (tool
   * calls, thinking, eval reports) via
   * `ConversationView`; replacements receive the same props.
   *
   * Props supplied by TrainingRunDetailPage:
   *   sample       - selected normalized sample object
   *   samples      - all samples in the selected rollout/prompt group
   *   trajectory   - sample.metadata.trajectory_messages[]
   *   rewardEvents - sample.reward_events[] (if emitted)
   *   rollout      - full expanded TrainingRolloutResult
   *   run          - current TrainingRun summary
   */
  import ConversationView from "./ConversationView.svelte";

  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
  } = $props();

  let messages = $derived(
    Array.isArray(trajectory) && trajectory.length
      ? trajectory
      : sample?.metadata?.trajectory_messages ?? null,
  );
</script>

<ConversationView
  {messages}
  response={sample?.response || ""}
  thinking={sample?.thinking || ""}
  evalReport={sample?.metadata?.eval_report}
/>
