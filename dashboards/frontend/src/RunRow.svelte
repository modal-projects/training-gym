<script>
  import { onDestroy } from "svelte";
  import { truncateId, fmtCluster, fmtLr } from "./lib/format.js";

  let { run, deployments = [] } = $props();

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function normalizePath(value) {
    return safeText(value).replace(/\/+$/, "");
  }

  function pathMatches(left, right) {
    const a = normalizePath(left);
    const b = normalizePath(right);
    if (!a || !b) return false;
    return a === b || a.startsWith(`${b}/`) || b.startsWith(`${a}/`);
  }

  let summary = $derived(run.config_summary || {});
  let result = $derived(run.train_result);
  let trainingRunId = $derived(result?.training_run_id || run.run_id || "");
  let modalAppUrl = $derived(run.modal_app_url || null);
  let wandbUrl = $derived(result?.wandb_url || summary.wandb_url || "");
  let wandbLinks = $derived(
    run.wandb_links?.length ? run.wandb_links : wandbUrl ? [{ label: "W&B", url: wandbUrl }] : [],
  );
  let resumeState = $derived(run.resume_state);
  let copiedTrainingRunId = $state(false);
  let copyResetTimer = null;
  let deployment = $derived(
    deployments.find((d) => {
      const deploymentAppName = safeText(d.app_name || "");
      const deploymentModelName = safeText(d.model_name || "");
      const deploymentModelPath = normalizePath(d.model_path || "");
      const deploymentCheckpointPath = normalizePath(d.checkpoint_path || "");
      const runModelName = safeText(result?.model_name || summary.model_name || "");
      const runModelPath = normalizePath(result?.model_path || "");
      const runCheckpointDir = normalizePath(result?.checkpoint_dir || "");

      if (run.deployment_id && deploymentAppName && run.deployment_id === deploymentAppName) {
        return true;
      }
      if (
        deploymentCheckpointPath &&
        (pathMatches(deploymentCheckpointPath, runCheckpointDir) ||
          pathMatches(deploymentCheckpointPath, runModelPath))
      ) {
        return true;
      }
      if (
        deploymentModelPath &&
        (pathMatches(deploymentModelPath, runCheckpointDir) ||
          pathMatches(deploymentModelPath, runModelPath))
      ) {
        return true;
      }
      return !!deploymentModelName && deploymentModelName === runModelName;
    }) || null,
  );

  function openModalApp() {
    if (!modalAppUrl) return;
    window.open(modalAppUrl, "_blank", "noopener,noreferrer");
  }

  function onRowKeydown(event) {
    if (!modalAppUrl) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openModalApp();
    }
  }

  async function copyTrainingRunId(event) {
    event.stopPropagation();
    if (!trainingRunId) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(trainingRunId);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = trainingRunId;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      copiedTrainingRunId = true;
      if (copyResetTimer) {
        clearTimeout(copyResetTimer);
      }
      copyResetTimer = setTimeout(() => {
        copiedTrainingRunId = false;
      }, 1200);
    } catch {
      copiedTrainingRunId = false;
    }
  }

  onDestroy(() => {
    if (copyResetTimer) clearTimeout(copyResetTimer);
  });
</script>

<tr
  class="framework-run-row"
  class:clickable={!!modalAppUrl}
  onclick={openModalApp}
  onkeydown={onRowKeydown}
  role={modalAppUrl ? "link" : undefined}
  tabindex={modalAppUrl ? "0" : undefined}
  aria-label={modalAppUrl ? `Open Modal app for training run ${run.run_id}` : undefined}
>
  <td class="[font-family:var(--font-mono)] text-[0.78rem] text-[color-mix(in_srgb,var(--accent)_78%,white)] cursor-default" title={run.run_id}>
    <div class="run-row-mono">{truncateId(run.run_id)}</div>
    {#if run.modal_app_id}
      <div class="text-(--muted-strong) text-[0.72rem] [font-family:var(--font-mono)] mt-[0.1rem]" title={run.modal_app_id}>{truncateId(run.modal_app_id)}</div>
    {/if}
  </td>
  <td class="whitespace-nowrap">
    {#if trainingRunId}
      <button
        type="button"
        class="inline-flex items-center gap-[0.35rem] p-[0.16rem_0.45rem] [border:1px_solid_var(--border)] rounded-[6px] bg-(--panel-alt) text-(--text) text-[0.66rem] cursor-copy hover:border-(--accent-border) hover:bg-[color-mix(in_srgb,var(--accent)_8%,transparent)] run-row-mono"
        title={`Click to copy ${trainingRunId}`}
        aria-label={`Copy training run id ${trainingRunId}`}
        onclick={copyTrainingRunId}
      >
        <span>{truncateId(trainingRunId)}</span>
        <span class="text-(--muted-strong) text-[0.64rem] uppercase tracking-[0.04em]">{copiedTrainingRunId ? "Copied" : "Copy"}</span>
      </button>
    {:else}
      —
    {/if}
  </td>
  <td>
    <div class="font-medium text-(--text-bright)">{summary.model_name || "—"}</div>
  </td>
  <td class="text-[0.75rem] whitespace-nowrap text-(--text)">{fmtCluster(summary)}</td>
  <td class="flex flex-wrap gap-[0.25rem]">
    {#if summary.lr}
      <span class="config-tag">lr {fmtLr(summary.lr)}</span>
    {/if}
    {#if summary.global_batch_size}
      <span class="config-tag">bs {summary.global_batch_size}</span>
    {/if}
    {#if summary.wandb_group}
      <span class="config-tag">{summary.wandb_group}</span>
    {/if}
  </td>
  <td class="whitespace-nowrap">
    {#if result}
      <span class="result-badge border-(--accent-border) bg-(--accent-soft) text-(--green)">Completed</span>
      <div class="result-meta run-row-mono" title={result.training_run_id}>
        TrainResult {truncateId(result.training_run_id)}
      </div>
      {#if result.checkpoint_dir}
        <div class="result-meta run-row-mono" title={result.checkpoint_dir}>
          {truncateId(result.checkpoint_dir)}
        </div>
      {/if}
      {#if resumeState}
        <div class="result-meta text-(--yellow)!">
          {resumeState.attempt_count > 1 ? `attempt ${resumeState.attempt_count}` : ""}
          {resumeState.resumed_from_checkpoint ? " resumed" : ""}
        </div>
      {/if}
    {:else if run.status === "stopped" || run.status === "failed"}
      <span class="result-badge border-[color-mix(in_srgb,#f87171_45%,transparent)] bg-[color-mix(in_srgb,#f87171_10%,transparent)] text-[#f87171]">No result</span>
    {:else}
      <span class="result-badge result-pending">Pending</span>
    {/if}
  </td>
  <td class="whitespace-nowrap">
    {#if deployment}
      <div class="font-medium text-(--text-bright) mb-[0.15rem]">
        {deployment.app_name || deployment.served_model_name || deployment.model_name || "Deployment"}
      </div>
      {#if deployment.url}
        <a
          class="pill-link text-(--green) [border:1px_solid_var(--accent-border)] bg-(--accent-soft) hover:bg-[color-mix(in_srgb,var(--accent)_18%,transparent)]"
          href={deployment.url}
          target="_blank"
          rel="noopener noreferrer"
          onclick={(event) => event.stopPropagation()}>Endpoint</a
        >
      {/if}
    {:else}
      <span class="result-badge result-pending">Not deployed</span>
    {/if}
  </td>
  <td>
    {#if modalAppUrl}
      <a
        class="pill-link text-(--accent) [border:1px_solid_var(--accent-border)] bg-(--accent-soft) hover:bg-[color-mix(in_srgb,var(--accent)_18%,transparent)]"
        href={modalAppUrl}
        target="_blank"
        rel="noopener noreferrer"
        onclick={(event) => event.stopPropagation()}>Modal</a
      >
    {/if}
    {#each wandbLinks as link (link.url)}
      <a
        class="pill-link text-(--yellow) [border:1px_solid_color-mix(in_srgb,var(--yellow)_45%,transparent)] bg-[color-mix(in_srgb,var(--yellow)_10%,transparent)] hover:bg-[color-mix(in_srgb,var(--yellow)_18%,transparent)]"
        href={link.url}
        target="_blank"
        rel="noopener noreferrer"
        onclick={(event) => event.stopPropagation()}>{link.label}</a
      >
    {/each}
    {#if result?.checkpoint_dir}
      <span class="tag" title={result.checkpoint_dir}>
        <strong>ckpt</strong>
      </span>
    {/if}
  </td>
</tr>
