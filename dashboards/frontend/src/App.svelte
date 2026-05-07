<script>
  import { onMount } from "svelte";
  import { Book, CheckCircle2, Rocket, Zap } from "lucide-svelte";
  import "./app.css";
  import Sidebar from "./components/Sidebar.svelte";
  import DashboardHeader from "./components/DashboardHeader.svelte";
  import TrainingPage from "./pages/TrainingPage.svelte";
  import DeploymentsPage from "./pages/DeploymentsPage.svelte";
  import EvalsPage from "./pages/EvalsPage.svelte";
  import { fetchRuns, fetchEvals, fetchDeployments } from "./lib/api.js";
  import logoSvg from "./lib/logo.svg";
  import { fmtDuration, fmtCluster, truncateId } from "./lib/format.js";

  const DOCS_URL = "https://gym.modal.dev";

  let allRuns = $state([]);
  let allEvals = $state([]);
  let allDeployments = $state([]);
  let loading = $state(true);
  let loadingEvals = $state(true);
  let loadingDeployments = $state(true);
  let error = $state(null);
  let search = $state("");
  let activeFrameworks = $state(new Set());
  let activeStatuses = $state(new Set());
  let activePage = $state("training");
  let loadRequestId = 0;

  const pageMeta = {
    training: { title: "Training runs" },
    deployments: { title: "Deployments" },
    evals: { title: "Evals" },
  };

  const pagePaths = {
    training: "/training",
    deployments: "/deployments",
    evals: "/evals",
  };

  function pageFromPath(pathname) {
    if (pathname === "/" || pathname.startsWith("/training")) return "training";
    if (pathname.startsWith("/deployments")) return "deployments";
    if (pathname.startsWith("/evals")) return "evals";
    return "training";
  }

  const navItems = [
    { key: "training", label: "Training runs", Icon: Zap, path: pagePaths.training },
    { key: "deployments", label: "Deployments", Icon: Rocket, path: pagePaths.deployments },
    { key: "evals", label: "Evals", Icon: CheckCircle2, path: pagePaths.evals },
  ];

  if (typeof window !== "undefined") {
    activePage = pageFromPath(window.location.pathname);
  }

  onMount(() => {
    const syncPageWithPath = () => {
      activePage = pageFromPath(window.location.pathname);
    };

    if (window.location.pathname === "/") {
      window.history.replaceState({}, "", pagePaths.training);
    } else {
      syncPageWithPath();
    }

    window.addEventListener("popstate", syncPageWithPath);
    return () => window.removeEventListener("popstate", syncPageWithPath);
  });

  function getFramework(run) {
    return run.framework || "(untagged)";
  }

  function getStatus(run) {
    if (run.train_result) return "Completed";
    if (run.status === "stopped") return "Stopped";
    if (run.status === "failed") return "Failed";
    return "Running";
  }

  function modelName(run) {
    return run.train_result?.model_name || run.config_summary?.model_name || "—";
  }

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function includesText(value, query) {
    return safeText(value).toLowerCase().includes(query);
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

  function getErrorMessage(value) {
    if (value instanceof Error) return value.message;
    if (typeof value === "string") return value;
    return "unknown error";
  }

  function fetchWithTimeout(fn, timeoutMs, label) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    return fn({ signal: controller.signal })
      .then((value) => {
        clearTimeout(timeoutId);
        return value;
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        if (err.name === "AbortError")
          throw new Error(`${label} request timed out after ${timeoutMs}ms`);
        throw err;
      });
  }

  function deploymentLabel(deployment) {
    return (
      deployment?.app_name ||
      deployment?.served_model_name ||
      deployment?.model_name ||
      "Deployment"
    );
  }

  function findRunForDeployment(deployment) {
    const deploymentAppName = safeText(deployment?.app_name || "");
    const deploymentModelName = safeText(deployment?.model_name || "");
    const deploymentModelPath = normalizePath(deployment?.model_path || "");
    const deploymentCheckpointPath = normalizePath(deployment?.checkpoint_path || "");

    return (
      allRuns.find((run) => {
        const result = run.train_result || {};
        const runModelName = safeText(
          result.model_name || run.config_summary?.model_name || "",
        );
        const runModelPath = normalizePath(result.model_path || "");
        const runCheckpointDir = normalizePath(result.checkpoint_dir || "");

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
      }) || null
    );
  }

  async function load() {
    const requestId = ++loadRequestId;
    const isStale = () => requestId !== loadRequestId;

    loading = true;
    loadingEvals = true;
    loadingDeployments = true;
    error = null;

    const evalsPromise = fetchWithTimeout(fetchEvals, 15000, "evals");
    const deploymentsPromise = fetchWithTimeout(fetchDeployments, 15000, "deployments");

    evalsPromise
      .then((evals) => {
        if (isStale()) return;
        allEvals = evals;
      })
      .catch((reason) => {
        if (isStale()) return;
        allEvals = [];
        console.warn(getErrorMessage(reason));
      })
      .finally(() => {
        if (isStale()) return;
        loadingEvals = false;
      });

    deploymentsPromise
      .then((deployments) => {
        if (isStale()) return;
        allDeployments = deployments;
      })
      .catch((reason) => {
        if (isStale()) return;
        allDeployments = [];
        console.warn(getErrorMessage(reason));
      })
      .finally(() => {
        if (isStale()) return;
        loadingDeployments = false;
      });

    try {
      const runs = await fetchWithTimeout(fetchRuns, 30000, "runs");
      if (isStale()) return;
      allRuns = runs;
      activeFrameworks = new Set(allRuns.map(getFramework));
      activeStatuses = new Set(allRuns.map(getStatus));
    } catch (e) {
      if (isStale()) return;
      error = getErrorMessage(e);
      allRuns = [];
      activeFrameworks = new Set();
      activeStatuses = new Set();
    }
    if (isStale()) return;
    loading = false;
  }

  load();

  let frameworks = $derived([...new Set(allRuns.map(getFramework))].sort());
  let statuses = $derived([...new Set(allRuns.map(getStatus))].sort());

  let fwCounts = $derived(
    allRuns.reduce((acc, run) => {
      const fw = getFramework(run);
      acc[fw] = (acc[fw] || 0) + 1;
      return acc;
    }, {}),
  );

  let statusCounts = $derived(
    allRuns.reduce((acc, run) => {
      const status = getStatus(run);
      acc[status] = (acc[status] || 0) + 1;
      return acc;
    }, {}),
  );

  let filteredRuns = $derived(
    allRuns
      .filter((run) => {
        if (!activeFrameworks.has(getFramework(run))) return false;
        if (!activeStatuses.has(getStatus(run))) return false;
        if (search) {
          const q = search.toLowerCase();
          if (
            !includesText(run.run_id, q) &&
            !includesText(run.modal_app_id, q) &&
            !includesText(run.config_summary?.model_name, q) &&
            !includesText(run.train_result?.training_run_id, q) &&
            !includesText(run.train_result?.checkpoint_dir, q) &&
            !includesText(run.train_result?.model_name, q) &&
            !includesText(run.train_result?.model_path, q) &&
            !includesText(run.deployment_id, q)
          ) {
            return false;
          }
        }
        return true;
      })
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
  );

  let completedTotal = $derived(allRuns.filter((run) => run.train_result).length);
  let stoppedTotal = $derived(allRuns.filter((run) => getStatus(run) === "Stopped").length);
  let failedTotal = $derived(allRuns.filter((run) => getStatus(run) === "Failed").length);
  let runningTotal = $derived(allRuns.length - completedTotal - stoppedTotal - failedTotal);

  let deploymentRows = $derived.by(() =>
    [...allDeployments]
      .map((deployment) => ({
        deployment,
        run: findRunForDeployment(deployment),
      }))
      .sort((a, b) =>
        deploymentLabel(a.deployment).localeCompare(
          deploymentLabel(b.deployment),
        ),
      ),
  );

  let linkedDeployments = $derived(
    deploymentRows.filter(({ run }) => run != null).length,
  );

  let distinctDeploymentModels = $derived(
    new Set(
      allDeployments.map(
        (deployment) =>
          deployment.model_name || deployment.served_model_name || "—",
      ),
    ).size,
  );

  function evalAccuracy(ev) {
    const rows = ev.rows || [];
    if (!rows.length) return 0;
    return rows.reduce((sum, row) => sum + (row.score || 0), 0) / rows.length;
  }

  function normalizeConfigValue(value) {
    if (value && typeof value === "object" && "value" in value) {
      return normalizeConfigValue(value.value);
    }
    if (Array.isArray(value)) {
      return value.map((item) => normalizeConfigValue(item));
    }
    if (value && typeof value === "object") {
      return Object.keys(value)
        .sort()
        .reduce((acc, key) => {
          acc[key] = normalizeConfigValue(value[key]);
          return acc;
        }, {});
    }
    return value ?? null;
  }

  function evalConfigKey(ev) {
    return JSON.stringify(normalizeConfigValue(ev.config || {}));
  }

  function evalConfigMeta(config) {
    const dataset =
      safeText(config?.dataset?.name) ||
      safeText(config?.dataset?.hf_repo) ||
      safeText(config?.dataset?.prompt_data) ||
      "—";
    const model =
      safeText(config?.deployment?.model_name) ||
      safeText(config?.deployment?.served_model_name) ||
      safeText(config?.model?.model_name) ||
      "—";
    const split = safeText(config?.dataset?.split);
    const judge =
      safeText(config?.judge?.model_name) ||
      safeText(config?.judge_model_name) ||
      "";
    return { dataset, model, split, judge };
  }

  let sortedEvals = $derived(
    [...allEvals].sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
  );

  let evalConfigGroups = $derived.by(() => {
    const groups = new Map();
    for (const ev of sortedEvals) {
      const key = evalConfigKey(ev);
      if (!groups.has(key)) {
        groups.set(key, {
          key,
          config: ev.config || {},
          runs: [],
          latestCreatedAt: 0,
        });
      }
      const group = groups.get(key);
      const avgScore = evalAccuracy(ev);
      const totalRows = (ev.rows || []).length;
      group.runs.push({
        eval: ev,
        avgScore,
        totalRows,
      });
      group.latestCreatedAt = Math.max(group.latestCreatedAt, ev.created_at || 0);
    }

    return [...groups.values()]
      .map((group) => {
        const sortedRuns = [...group.runs].sort(
          (a, b) =>
            b.avgScore - a.avgScore ||
            (b.eval.created_at || 0) - (a.eval.created_at || 0),
        );
        const totalEvals = sortedRuns.length;
        const totalExamples = sortedRuns.reduce(
          (sum, run) => sum + run.totalRows,
          0,
        );
        const weightedScoreTotal = sortedRuns.reduce(
          (sum, run) => sum + run.avgScore * run.totalRows,
          0,
        );
        const bestScore = sortedRuns[0]?.avgScore ?? 0;
        const avgAccuracy =
          totalExamples > 0
            ? weightedScoreTotal / totalExamples
            : totalEvals > 0
              ? sortedRuns.reduce((sum, run) => sum + run.avgScore, 0) / totalEvals
              : 0;
        return {
          ...group,
          meta: evalConfigMeta(group.config),
          bestScore,
          totalEvals,
          avgAccuracy,
          runs: sortedRuns.map((entry, idx) => ({
            ...entry,
            rank: idx + 1,
            deltaFromBest: entry.avgScore - bestScore,
          })),
        };
      })
      .sort(
        (a, b) =>
          (b.latestCreatedAt || 0) - (a.latestCreatedAt || 0) ||
          b.runs.length - a.runs.length,
      );
  });

  let evalConfigCount = $derived(evalConfigGroups.length);

  let avgAccuracy = $derived(
    allEvals.length
      ? allEvals.reduce((sum, ev) => sum + evalAccuracy(ev), 0) / allEvals.length
      : 0,
  );

  let statusText = $derived.by(() => {
    if (activePage === "training" && loading) return "loading...";
    if (activePage === "evals" && loadingEvals) return "loading...";
    if (activePage === "deployments" && loadingDeployments) return "loading...";
    if (error) return "error";
    if (activePage === "evals")
      return `${allEvals.length} eval${allEvals.length === 1 ? "" : "s"}`;
    if (activePage === "deployments")
      return `${allDeployments.length} deployment${allDeployments.length === 1 ? "" : "s"}`;
    if (!allRuns.length) return "0 runs";
    return `${filteredRuns.length} of ${allRuns.length} runs`;
  });

  function toggleFramework(framework) {
    const next = new Set(activeFrameworks);
    if (next.has(framework)) next.delete(framework);
    else next.add(framework);
    activeFrameworks = next;
  }

  function toggleAllFrameworks() {
    if (activeFrameworks.size === frameworks.length) activeFrameworks = new Set();
    else activeFrameworks = new Set(frameworks);
  }

  function toggleStatus(status) {
    const next = new Set(activeStatuses);
    if (next.has(status)) next.delete(status);
    else next.add(status);
    activeStatuses = next;
  }

  function setActivePage(page) {
    activePage = page;
    if (typeof window === "undefined") return;
    const targetPath = pagePaths[page] || pagePaths.training;
    if (window.location.pathname !== targetPath) {
      window.history.pushState({}, "", targetPath);
    }
  }

  function openTrainingRun(runId) {
    search = runId;
    setActivePage("training");
  }
</script>

<div class="app-shell">
  <header class="top-navbar">
    <div class="top-brand">
      <img src={logoSvg} alt="Modal" class="top-brand-logo" />
      <span class="top-brand-title">
        <span class="top-brand-modal">Modal</span>
        <span class="top-brand-gym">Training Gym</span>
      </span>
    </div>
    <a
      class="docs-button"
      href={DOCS_URL}
      target="_blank"
      rel="noopener noreferrer"
    >
      <Book size={14} strokeWidth={2.1} />
      <span>Docs</span>
    </a>
  </header>

  <div class="shell">
    <Sidebar {navItems} {activePage} onNavigate={setActivePage} />

    <main class="workspace">
      <DashboardHeader title={pageMeta[activePage].title} {statusText} onRefresh={load} />

    {#if activePage === "training"}
      <TrainingPage
        {allRuns}
        {completedTotal}
        {runningTotal}
        {stoppedTotal}
        {failedTotal}
        {frameworks}
        {fwCounts}
        {activeFrameworks}
        {statuses}
        {statusCounts}
        {activeStatuses}
        {filteredRuns}
        {loading}
        {error}
        {modelName}
        {getStatus}
        {fmtCluster}
        {fmtDuration}
        bind:search
        onToggleFramework={toggleFramework}
        onToggleAllFrameworks={toggleAllFrameworks}
        onToggleStatus={toggleStatus}
      />
    {:else if activePage === "deployments"}
      <DeploymentsPage
        {allDeployments}
        {linkedDeployments}
        {distinctDeploymentModels}
        loading={loadingDeployments}
        {error}
        {deploymentRows}
        {deploymentLabel}
        {truncateId}
        onOpenTrainingRun={openTrainingRun}
      />
    {:else if activePage === "evals"}
      <EvalsPage
        {allEvals}
        {avgAccuracy}
        {evalConfigCount}
        loading={loadingEvals}
        {error}
        {evalConfigGroups}
      />
    {/if}
    </main>
  </div>
</div>

<style>
  .app-shell {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
    background: var(--bg);
  }

  .top-navbar {
    border-bottom: 1px solid var(--border);
    background: var(--bg-depth);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 18px 1.35rem;
  }

  .top-brand {
    display: inline-flex;
    align-items: center;
    gap: 0.62rem;
    min-width: 0;
  }

  .top-brand-logo {
    width: 1.7rem;
    height: 1.7rem;
    flex: 0 0 auto;
  }

  .top-brand-title {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    font-size: 1.85rem;
    line-height: 1;
    padding-block: 0.08rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    white-space: nowrap;
  }

  .top-brand-modal {
    color: var(--text-bright);
  }

  .top-brand-gym {
    color: var(--green);
  }

  .docs-button {
    border: 0;
    border-radius: 8px;
    color: var(--muted);
    background: transparent;
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.34rem 0.68rem;
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    flex: 0 0 auto;
  }

  .docs-button:hover {
    color: var(--text-bright);
    background: color-mix(in srgb, white 4%, transparent);
  }

  .shell {
    display: grid;
    grid-template-columns: 232px minmax(0, 1fr);
    min-height: 0;
    background: var(--bg);
  }

  .workspace {
    padding: 0.8rem 1.5rem 1.5rem;
  }

  @media (max-width: 900px) {
    .top-navbar {
      padding: 14px 0.95rem;
    }

    .top-brand-logo {
      width: 1.35rem;
      height: 1.35rem;
    }

    .top-brand-title {
      font-size: 1.35rem;
    }

    .shell {
      grid-template-columns: 1fr;
    }

    .workspace {
      padding: 1rem;
    }
  }
</style>
