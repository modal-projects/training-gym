<script>
  import { onMount } from "svelte";
  import { FlaskConical, Rocket, Zap } from "lucide-svelte";
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
  let error = $state(null);
  let search = $state("");
  let activeFrameworks = $state(new Set());
  let activeStatuses = $state(new Set());
  let activePage = $state("training");

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
    { key: "evals", label: "Evals", Icon: FlaskConical, path: pagePaths.evals },
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
    return run.train_result ? "Completed" : "Pending";
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
    loading = true;
    error = null;
    try {
      const [runs, evals, deployments] = await Promise.all([
        fetchRuns(),
        fetchEvals(),
        fetchDeployments(),
      ]);
      allRuns = runs;
      allEvals = evals;
      allDeployments = deployments;
      activeFrameworks = new Set(allRuns.map(getFramework));
      activeStatuses = new Set(allRuns.map(getStatus));
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
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
  let pendingTotal = $derived(allRuns.length - completedTotal);

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
    if (loading) return "loading...";
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

<div class="shell">
  <Sidebar
    logoSrc={logoSvg}
    {navItems}
    {activePage}
    onNavigate={setActivePage}
  />

  <main class="workspace">
    <DashboardHeader
      title={pageMeta[activePage].title}
      {statusText}
      docsUrl={DOCS_URL}
      onRefresh={load}
    />

    {#if activePage === "training"}
      <TrainingPage
        {allRuns}
        {completedTotal}
        {pendingTotal}
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
        {loading}
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
        {loading}
        {error}
        {evalConfigGroups}
      />
    {/if}
  </main>
</div>

<style>
  .shell {
    display: grid;
    grid-template-columns: 232px minmax(0, 1fr);
    min-height: 100vh;
    background: var(--bg);
  }

  .workspace {
    padding: 0.8rem 1.5rem 1.5rem;
  }

  @media (max-width: 900px) {
    .shell {
      grid-template-columns: 1fr;
    }

    .workspace {
      padding: 1rem;
    }
  }
</style>
