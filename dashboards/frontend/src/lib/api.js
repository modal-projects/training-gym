const SERVER = "/api";

function safeStr(v) {
  if (v && typeof v === "object" && "value" in v) return v.value;
  return v != null ? String(v) : "";
}

function safeTimestamp(v) {
  if (v && typeof v === "object" && "value" in v) return safeTimestamp(v.value);
  if (typeof v === "number") return Number.isFinite(v) ? v : 0;
  if (typeof v === "string" && v.trim()) {
    const parsed = Number(v);
    if (Number.isFinite(parsed)) return parsed;
    const epochMs = Date.parse(v);
    if (Number.isFinite(epochMs)) return Math.floor(epochMs / 1000);
  }
  return 0;
}

function modalAppUrl(modalAppId) {
  const appId = safeStr(modalAppId).trim();
  if (!appId) return null;
  if (appId.startsWith("http://") || appId.startsWith("https://")) return appId;
  return `https://modal.com/id/${appId}`;
}

function extractConfigSummary(config) {
  if (!config || typeof config !== "object") return {};
  const model = config.model || {};
  const preset = config.preset || {};
  const recipe = config.recipe || {};
  const compute = Object.keys(recipe).length ? recipe : preset;
  const wandb = config.wandb || {};
  const dataset = config.dataset || {};
  const datasetName =
    safeStr(dataset.hf_repo) ||
    safeStr(dataset.prompt_data) ||
    safeStr(dataset.name);
  return {
    model_name: safeStr(model.model_name || ""),
    gpu_type: safeStr(compute.gpu_type || ""),
    actor_num_nodes: compute.actor_num_nodes || 0,
    actor_num_gpus_per_node: compute.actor_num_gpus_per_node || 0,
    lr: config.lr || 0,
    global_batch_size: config.global_batch_size || 0,
    wandb_project: (wandb.project) || "",
    wandb_group: (wandb.group) || "",
    dataset_name: datasetName,
    dataset_prompt_data: safeStr(dataset.prompt_data || ""),
  };
}

function summarizeDeployment(d) {
  const config = d.deployment_config || {};
  const model = config.model || {};
  const checkpoint = config.checkpoint || {};
  const health = d.health || {};
  const appId = safeStr(d.modal_app_id || "");
  const rawStatus =
    safeStr(d.status || "") ||
    safeStr(d.deployment_status || "") ||
    safeStr(d.state || "") ||
    safeStr(d.health_status || "") ||
    safeStr(health.status || "") ||
    safeStr(config.status || "") ||
    safeStr(config.state || "");
  return {
    deployment_id: safeStr(d.deployment_id || ""),
    app_name: safeStr(config.app_name || ""),
    served_model_name: safeStr(config.served_model_name || ""),
    model_name: safeStr(model.model_name || ""),
    model_path: safeStr(model.model_path || ""),
    checkpoint_path: safeStr(checkpoint.path || ""),
    status: rawStatus,
    version: safeStr(d.version || d.deployment_version || ""),
    created_at: safeTimestamp(d.created_at || d.updated_at || d.last_updated_at),
    url: safeStr(d.url || ""),
    modal_app_id: appId,
    modal_app_url: modalAppUrl(appId),
  };
}

function summarizeTrainResult(r) {
  const wandbRunId = r.wandb_training_run_id || "";
  const wandbProject = r.wandb_project || "";
  const wandbEntity = r.wandb_entity || "";
  let wandb_url = null;
  if (wandbRunId && wandbProject && wandbEntity) {
    wandb_url = `https://wandb.ai/${wandbEntity}/${wandbProject}/runs/${wandbRunId}`;
  }
  return {
    training_run_id: r.training_run_id || "",
    app_name: r.app_name || "",
    checkpoint_dir: r.checkpoint_dir || "",
    model_name: safeStr(r.model_config?.model_name || ""),
    model_path: safeStr(r.model_config?.model_path || ""),
    wandb_url,
    wandb_project: wandbProject,
    wandb_entity: wandbEntity,
    wandb_training_run_id: wandbRunId,
  };
}

export async function fetchRuns({ signal } = {}) {
  const [runsRes, resultsRes] = await Promise.all([
    fetch(`${SERVER}/runs`, { signal }),
    fetch(`${SERVER}/train-results`, { signal }),
  ]);
  if (!runsRes.ok) throw new Error(`HTTP ${runsRes.status}`);
  const runs = await runsRes.json();
  const trainResults = resultsRes.ok ? await resultsRes.json() : [];

  const resultsByRunId = {};
  for (const r of trainResults) {
    const id = r.training_run_id || "";
    if (id) resultsByRunId[id] = r;
  }

  const normalizedRuns = runs.map((run, index) => {
    const rawRunId = safeStr(run.run_id || run.training_run_id);
    const modalAppId = safeStr(run.modal_app_id);
    const runId =
      rawRunId ||
      `unknown-run-${modalAppId || "no-app"}-${safeTimestamp(run.created_at || run.started_at)}-${index}`;
    const trainResult = resultsByRunId[rawRunId] || null;
    const startedAt = run.started_at || run.created_at || 0;
    const endedAt = run.ended_at || null;
    const completedAt =
      run.completed_at || (run.status === "completed" ? endedAt : null);
    const durationSeconds =
      typeof run.duration_seconds === "number" && run.duration_seconds >= 0
        ? run.duration_seconds
        : startedAt && endedAt
          ? Math.max(0, endedAt - startedAt)
          : null;
    return {
      training_run_id: runId,
      run_id: runId,
      modal_app_id: modalAppId,
      modal_app_url: modalAppUrl(modalAppId),
      framework: safeStr(run.framework) || "(untagged)",
      status: run.status || "running",
      dataset_id: safeStr(run.dataset_id || ""),
      deployment_id: safeStr(run.deployment_id || ""),
      config: run.config || {},
      config_summary: extractConfigSummary(run.config),
      created_at: run.created_at || 0,
      started_at: startedAt,
      ended_at: endedAt,
      completed_at: completedAt,
      duration_seconds: durationSeconds,
      metadata: run.metadata || null,
      has_train_result: !!trainResult,
      train_result: trainResult ? summarizeTrainResult(trainResult) : null,
    };
  });

  const dedupedByRunId = new Map();
  for (const run of normalizedRuns) {
    const existing = dedupedByRunId.get(run.run_id);
    if (!existing) {
      dedupedByRunId.set(run.run_id, run);
      continue;
    }
    if (run.train_result && !existing.train_result) {
      dedupedByRunId.set(run.run_id, run);
      continue;
    }
    if (existing.train_result && !run.train_result) continue;
    const existingCreatedAt = safeTimestamp(existing.created_at || existing.started_at);
    const incomingCreatedAt = safeTimestamp(run.created_at || run.started_at);
    if (incomingCreatedAt >= existingCreatedAt) dedupedByRunId.set(run.run_id, run);
  }

  return [...dedupedByRunId.values()];
}

export async function fetchEvals({ signal } = {}) {
  const res = await fetch(`${SERVER}/evals`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const evals = await res.json();
  const seen = new Set();
  return evals.filter((e) => {
    if (seen.has(e.eval_id)) return false;
    seen.add(e.eval_id);
    return true;
  });
}

export async function fetchDeployments({ signal } = {}) {
  const res = await fetch(`${SERVER}/deployments`, { signal });
  if (!res.ok) return [];
  const deployments = await res.json();
  return deployments.map(summarizeDeployment);
}

export async function fetchTrainResult(trainingRunId) {
  const res = await fetch(
    `${SERVER}/train-results/${encodeURIComponent(trainingRunId)}`
  );
  if (!res.ok) return null;
  return await res.json();
}

export async function fetchEvalDetail(evalId) {
  const res = await fetch(
    `${SERVER}/evals/${encodeURIComponent(evalId)}`
  );
  if (!res.ok) return null;
  return await res.json();
}
