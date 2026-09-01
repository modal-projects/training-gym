<script>
  import { onMount } from "svelte";
  import { Book, CheckCircle2, Zap } from "lucide-svelte";
  import "./app.css";
  import Sidebar from "./components/Sidebar.svelte";
  import DashboardHeader from "./components/DashboardHeader.svelte";
  import TrainingPage from "./pages/TrainingPage.svelte";
  import TrainingRunDetailPage from "./pages/TrainingRunDetailPage.svelte";
  import EvalsPage from "./pages/EvalsPage.svelte";
  import { fetchRuns, fetchEvals, fetchEvalDetail } from "./lib/api.js";
  import logoSvg from "./lib/logo.svg";
  import { fmtDuration } from "./lib/format.js";

  const DOCS_URL = "https://gym.modal.dev";

  let allRuns = $state([]);
  let allEvals = $state([]);
  let loading = $state(true);
  let loadingEvals = $state(false);
  let error = $state(null);
  let search = $state("");
  let activeRecipes = $state(new Set());
  let activeStatuses = $state(new Set());
  let activeGroups = $state(new Set());
  let trainingGroupBy = $state("none");
  // Recipe/status/group values we've seen across loads. New ones are
  // auto-enabled in the filters once; the user's selections are never reset by
  // a refresh.
  let seenRecipes = new Set();
  let seenStatuses = new Set();
  let seenGroups = new Set();
  let activePage = $state("training");
  let activeTrainingRunId = $state(null);
  // When set (and no full detail page is open), the training list shows a
  // summary drawer for this run — set by "Collapse" on the detail page.
  let drawerRunId = $state(null);
  // True while any data fetch is in flight (manual or the 5s auto-refresh) —
  // drives the spinning refresh button. Distinct from `loading`, which only
  // gates the cold-start skeleton.
  let refreshing = $state(false);
  let runsRequestId = 0;
  let hasLoadedRuns = false;
  let initialRunsLoadStarted = false;
  let evalsRequestId = 0;
  let hasLoadedEvals = $state(false);

  const pageMeta = {
    training: { title: "Training runs" },
    evals: { title: "Evals" },
  };

  const pagePaths = {
    training: "/training",
    evals: "/evals",
  };

  function pageFromPath(pathname) {
    if (pathname === "/" || pathname.startsWith("/training")) return "training";
    if (pathname.startsWith("/evals")) return "evals";
    return "training";
  }

  function runIdFromPath(pathname) {
    if (!pathname.startsWith("/training/")) return null;
    const tail = pathname.slice("/training/".length).split("/")[0];
    return tail ? decodeURIComponent(tail) : null;
  }

  const navItems = [
    { key: "training", label: "Training runs", Icon: Zap, path: pagePaths.training },
    { key: "evals", label: "Evals", Icon: CheckCircle2, path: pagePaths.evals },
  ];

  if (typeof window !== "undefined") {
    activePage = pageFromPath(window.location.pathname);
    activeTrainingRunId = runIdFromPath(window.location.pathname);
  }

  onMount(() => {
    const syncPageWithPath = () => {
      activePage = pageFromPath(window.location.pathname);
      activeTrainingRunId = runIdFromPath(window.location.pathname);
    };

    if (window.location.pathname === "/") {
      window.history.replaceState({}, "", pagePaths.training);
    } else {
      syncPageWithPath();
    }

    window.addEventListener("popstate", syncPageWithPath);

    // Auto-refresh the active page's data every 5s so running training runs,
    // their status/stage and rollouts stay live. Current data stays on screen
    // (no skeleton) and only the refresh button spins while fetching. A run
    // detail page refreshes its own run, so skip the full list there.
    // A backgrounded tab refreshes nothing and catches up on the way back:
    // polling a hidden tab only burns memory and battery, and mobile browsers
    // discard tabs that keep working while off-screen.
    const refresh = window.setInterval(() => {
      if (document.hidden) return;
      if (activePage === "training" && activeTrainingRunId) return;
      void load();
    }, 5000);

    const onVisibilityChange = () => {
      if (document.hidden) return;
      if (activePage === "training" && activeTrainingRunId) return;
      void load();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      window.removeEventListener("popstate", syncPageWithPath);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.clearInterval(refresh);
    };
  });

  function getRecipe(run) {
    return run.recipe || run.framework || "(untagged)";
  }

  const NO_GROUP = "(no group)";

  function getGroup(run) {
    return safeText(run.group_id) || NO_GROUP;
  }

  function getStatus(run) {
    return safeText(run.display_status) || "pending";
  }

  function getTrainingRunStatus(run) {
    return safeText(run.status).toLowerCase();
  }

  function getFrameworkStatus(run) {
    return safeText(run.framework_status);
  }

  function showFrameworkStatus(run) {
    if (getTrainingRunStatus(run) === "running") return true;
    return !!run.framework_status;
  }

  function modelName(run) {
    return run.model || "—";
  }

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function includesText(value, query) {
    return safeText(value).toLowerCase().includes(query);
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

  async function loadRuns() {
    const requestId = ++runsRequestId;
    const isStale = () => requestId !== runsRequestId;

    // Skeleton only until the first response settles. Once we've completed a
    // load attempt (success or failure), refreshes keep the current rows on
    // screen — the spinning refresh button is the only "loading" affordance.
    if (!hasLoadedRuns) loading = true;
    error = null;

    try {
      const runs = await fetchWithTimeout(fetchRuns, 30000, "runs");
      if (isStale()) return;
      allRuns = runs;
      // Auto-enable newly-seen recipes/statuses without resetting the user's
      // current filter selection on every refresh.
      const nextRecipes = new Set(activeRecipes);
      const nextStatuses = new Set(activeStatuses);
      const nextGroups = new Set(activeGroups);
      let recipesChanged = false;
      let statusesChanged = false;
      let groupsChanged = false;
      for (const run of allRuns) {
        const recipe = getRecipe(run);
        if (!seenRecipes.has(recipe)) {
          seenRecipes.add(recipe);
          nextRecipes.add(recipe);
          recipesChanged = true;
        }
        const status = getStatus(run);
        if (!seenStatuses.has(status)) {
          seenStatuses.add(status);
          nextStatuses.add(status);
          statusesChanged = true;
        }
        const group = getGroup(run);
        if (!seenGroups.has(group)) {
          seenGroups.add(group);
          nextGroups.add(group);
          groupsChanged = true;
        }
      }
      if (recipesChanged) activeRecipes = nextRecipes;
      if (statusesChanged) activeStatuses = nextStatuses;
      if (groupsChanged) activeGroups = nextGroups;
    } catch (e) {
      if (isStale()) return;
      // Keep the data we already have on a transient refresh failure — only
      // surface the error (and clear) when there's nothing to show yet.
      // Otherwise the page flickers to "Loading…"/empty on every flaky poll.
      if (!allRuns.length) {
        error = getErrorMessage(e);
        activeRecipes = new Set();
        activeStatuses = new Set();
        activeGroups = new Set();
      }
    } finally {
      // Always retire the cold-start skeleton once any attempt settles — even a
      // stale one. A slow request superseded by the 5s auto-refresh must not
      // leave `loading` pinned true forever.
      hasLoadedRuns = true;
      loading = false;
    }
  }

  async function loadEvals() {
    const requestId = ++evalsRequestId;
    const isStale = () => requestId !== evalsRequestId;

    if (!allEvals.length) loadingEvals = true;
    try {
      const evals = await fetchWithTimeout(fetchEvals, 15000, "evals");
      if (isStale()) return;
      allEvals = evals;
      hasLoadedEvals = true;
    } catch (reason) {
      if (isStale()) return;
      if (!allEvals.length) allEvals = [];
      console.warn(getErrorMessage(reason));
    }
    if (!isStale()) loadingEvals = false;
  }

  async function load() {
    // One refresh at a time. The runs payload can take longer to arrive than
    // the 5s interval, and overlapping fetches stack whole copies of it in
    // memory for a response that gets thrown away as stale anyway.
    if (refreshing) return;
    refreshing = true;
    try {
      const tasks = [loadRuns()];
      if (activePage === "evals") {
        tasks.push(loadEvals());
      }
      await Promise.all(tasks);
    } finally {
      refreshing = false;
    }
  }

  $effect(() => {
    if (
      !activeTrainingRunId &&
      !hasLoadedRuns &&
      !initialRunsLoadStarted
    ) {
      initialRunsLoadStarted = true;
      void loadRuns();
    } else if (activeTrainingRunId && !hasLoadedRuns) {
      loading = false;
    }
    if (activePage === "evals" && !hasLoadedEvals) {
      void loadEvals();
    }
  });

  let recipes = $derived([...new Set(allRuns.map(getRecipe))].sort());
  let statuses = $derived([...new Set(allRuns.map(getStatus))].sort());
  // Real group ids first (alphabetical), with "(no group)" pinned last so the
  // sweep groups are what you see at the top of the filter.
  let groups = $derived(
    [...new Set(allRuns.map(getGroup))].sort((a, b) => {
      if (a === NO_GROUP) return 1;
      if (b === NO_GROUP) return -1;
      return a.localeCompare(b);
    }),
  );

  let recipeCounts = $derived(
    allRuns.reduce((acc, run) => {
      const recipe = getRecipe(run);
      acc[recipe] = (acc[recipe] || 0) + 1;
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

  let groupCounts = $derived(
    allRuns.reduce((acc, run) => {
      const group = getGroup(run);
      acc[group] = (acc[group] || 0) + 1;
      return acc;
    }, {}),
  );

  let filteredRuns = $derived(
    allRuns
      .filter((run) => {
        if (!activeRecipes.has(getRecipe(run))) return false;
        if (!activeStatuses.has(getStatus(run))) return false;
        if (!activeGroups.has(getGroup(run))) return false;
        if (search) {
          const q = search.toLowerCase();
          if (
            !includesText(run.run_id, q) &&
            !includesText(run.modal_app_id, q) &&
            !includesText(run.group_id, q) &&
            !includesText(JSON.stringify(run.group_tags || {}), q) &&
            !includesText(run.model, q) &&
            !includesText(run.dataset, q) &&
            !includesText(run.train_result?.training_run_id, q) &&
            !includesText(run.train_result?.checkpoint_dir, q) &&
            !includesText(run.train_result?.model_name, q) &&
            !includesText(run.train_result?.model_path, q) &&
            !includesText(run.framework_status, q) &&
            !includesText(run.deployment_id, q)
          ) {
            return false;
          }
        }
        return true;
      })
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
  );

  const trainingGroupKeyFns = {
    group: getGroup,
    dataset: (run) => safeText(run.dataset) || "(no dataset)",
    model: modelName,
  };

  const trainingGroupKey = (run, groupBy) => trainingGroupKeyFns[groupBy]?.(run) ?? "";

  // Buckets inherit filteredRuns' recency sort: groups come out ordered by
  // newest member and runs stay sorted within each group.
  let trainingRunGroups = $derived.by(() => {
    if (trainingGroupBy === "none") return [];
    const buckets = Map.groupBy(filteredRuns, (run) => trainingGroupKey(run, trainingGroupBy));
    return [...buckets].map(([key, runs]) => ({
      key,
      runs,
      latestCreatedAt: runs[0]?.created_at || null,
    }));
  });

  let completedTotal = $derived(allRuns.filter((run) => getStatus(run) === "completed").length);
  let cancelledTotal = $derived(allRuns.filter((run) => getStatus(run) === "cancelled").length);
  let stoppedTotal = $derived(allRuns.filter((run) => getStatus(run) === "stopped").length);
  let failedTotal = $derived(allRuns.filter((run) => getStatus(run) === "failed").length);
  let runningTotal = $derived(
    allRuns.length - completedTotal - cancelledTotal - stoppedTotal - failedTotal,
  );

  function evalAccuracy(ev) {
    if (typeof ev.mean === "number") return ev.mean;
    const rows = ev.rows || [];
    if (!rows.length) return 0;
    return rows.reduce((sum, row) => sum + (row.score || 0), 0) / rows.length;
  }

  function evalCreatedAt(ev) {
    const raw = ev?.created_at;
    if (raw && typeof raw === "object" && "value" in raw) {
      return evalCreatedAt({ created_at: raw.value });
    }
    if (typeof raw === "number") return Number.isFinite(raw) ? raw : 0;
    const text = safeText(raw).trim();
    if (!text) return 0;
    const numeric = Number(text);
    if (Number.isFinite(numeric)) return numeric;
    const epochMs = Date.parse(text);
    if (Number.isFinite(epochMs)) return Math.floor(epochMs / 1000);
    return 0;
  }

  // Maps a raw eval status onto a coarse filter/count bucket
  // ("Completed" | "Pending" | "Failed"), the StatusPill color/icon variant,
  // and a human label for the four eval phases.
  function getEvalDisplay(ev) {
    const rawStatus = safeText(ev.status).toLowerCase();
    if (rawStatus === "deploying_model" || rawStatus === "deploying") {
      return { bucket: "Pending", pill: "running", label: "Deploying model" };
    }
    if (
      rawStatus === "running_eval" ||
      rawStatus === "running" ||
      rawStatus === "pending" ||
      rawStatus === "queued" ||
      rawStatus === "initializing"
    ) {
      return { bucket: "Pending", pill: "running", label: "Running eval" };
    }
    if (
      rawStatus === "completed" ||
      rawStatus === "success" ||
      rawStatus === "succeeded"
    ) {
      return { bucket: "Completed", pill: "completed", label: "Success" };
    }
    if (rawStatus === "failed" || rawStatus === "error") {
      return { bucket: "Failed", pill: "failed", label: "Failed" };
    }
    const total = ev.total ?? (Array.isArray(ev.rows) ? ev.rows.length : 0);
    if (total > 0) {
      return { bucket: "Completed", pill: "completed", label: "Success" };
    }
    return { bucket: "Pending", pill: "running", label: "Pending" };
  }

  function getEvalStatus(ev) {
    return getEvalDisplay(ev).bucket;
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

  function evalConfigMeta(config, ev = null) {
    const evalConfig = ev?.eval_config || {};
    const sourceConfig = ev?.config || {};
    const dataset =
      safeText(config?.dataset?.name) ||
      safeText(config?.dataset?.hf_repo) ||
      safeText(config?.dataset?.prompt_data) ||
      safeText(config?.dataset_name) ||
      safeText(sourceConfig?.dataset?.name) ||
      safeText(sourceConfig?.dataset?.hf_repo) ||
      safeText(sourceConfig?.dataset?.prompt_data) ||
      safeText(sourceConfig?.dataset_name) ||
      safeText(evalConfig?.dataset_name) ||
      safeText(ev?.dataset_name) ||
      "—";
    const model = safeText(ev?.model_name) || "—";
    const split =
      safeText(config?.dataset?.split) ||
      safeText(sourceConfig?.dataset?.split) ||
      safeText(evalConfig?.dataset?.split);
    const judge =
      safeText(config?.judge?.model_name) ||
      safeText(config?.judge_model_name) ||
      safeText(sourceConfig?.judge?.model_name) ||
      safeText(sourceConfig?.judge_model_name) ||
      safeText(evalConfig?.judge?.model_name) ||
      safeText(evalConfig?.judge_model_name) ||
      "";
    const evalFn =
      safeText(config?.eval_fn_name) ||
      safeText(config?.grader_name) ||
      safeText(sourceConfig?.eval_fn_name) ||
      safeText(sourceConfig?.grader_name) ||
      safeText(evalConfig?.eval_fn_name) ||
      safeText(evalConfig?.grader_name) ||
      safeText(ev?.eval_fn_name) ||
      "";
    return { dataset, model, split, judge, evalFn };
  }

  let sortedEvals = $derived(
    [...allEvals].sort((a, b) => evalCreatedAt(b) - evalCreatedAt(a)),
  );

  let evalConfigGroups = $derived.by(() => {
    const groups = new Map();
    for (const ev of sortedEvals) {
      const key = safeText(ev.eval_config_id).trim() || evalConfigKey(ev);
      if (!groups.has(key)) {
        groups.set(key, {
          key,
          evalConfigId: key,
          config: ev.config || {},
          runs: [],
          latestCreatedAt: 0,
        });
      }
      const group = groups.get(key);
      const createdAt = evalCreatedAt(ev);
      if (
        (!group.config || Object.keys(group.config).length === 0) &&
        ev.config &&
        Object.keys(ev.config).length > 0
      ) {
        group.config = ev.config;
      }
      const avgScore = evalAccuracy(ev);
      const totalRows = ev.total ?? (ev.rows || []).length;
      const display = getEvalDisplay(ev);
      group.runs.push({
        eval: ev,
        avgScore,
        totalRows,
        status: display.bucket,
        pillStatus: display.pill,
        statusLabel: display.label,
        createdAt,
      });
      group.latestCreatedAt = Math.max(group.latestCreatedAt, createdAt);
    }

    return [...groups.values()]
      .map((group) => {
        const sortedRuns = [...group.runs].sort(
          (a, b) =>
            b.createdAt - a.createdAt ||
            b.avgScore - a.avgScore,
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
        const completedCount = sortedRuns.filter(
          (run) => run.status === "Completed",
        ).length;
        const pendingCount = sortedRuns.filter(
          (run) => run.status === "Pending",
        ).length;
        const failedCount = sortedRuns.filter((run) => run.status === "Failed").length;
        return {
          ...group,
          meta: evalConfigMeta(group.config, sortedRuns[0]?.eval),
          bestScore,
          totalEvals,
          avgAccuracy,
          completedCount,
          pendingCount,
          failedCount,
          runs: sortedRuns,
        };
      })
      .sort(
        (a, b) =>
          (b.latestCreatedAt || 0) - (a.latestCreatedAt || 0) ||
          b.runs.length - a.runs.length,
      );
  });

  let evalCompletedTotal = $derived(
    allEvals.filter((ev) => getEvalStatus(ev) === "Completed").length,
  );
  let evalPendingTotal = $derived(
    allEvals.filter((ev) => getEvalStatus(ev) === "Pending").length,
  );
  let evalFailedTotal = $derived(
    allEvals.filter((ev) => getEvalStatus(ev) === "Failed").length,
  );
  let activeTrainingRun = $derived(
    allRuns.find((run) => run.run_id === activeTrainingRunId) || null,
  );

  let statusText = $derived.by(() => {
    if (activePage === "training" && activeTrainingRunId) return "run details";
    if (activePage === "training" && loading) return "loading...";
    if (activePage === "evals" && loadingEvals) return "loading...";
    if (error) return "error";
    if (activePage === "evals")
      return `${allEvals.length} eval${allEvals.length === 1 ? "" : "s"}`;
    if (!allRuns.length) return "0 runs";
    return `${filteredRuns.length} of ${allRuns.length} runs`;
  });

  function toggleRecipe(recipe) {
    const next = new Set(activeRecipes);
    if (next.has(recipe)) next.delete(recipe);
    else next.add(recipe);
    activeRecipes = next;
  }

  function selectAllRecipes() {
    activeRecipes = new Set(recipes);
  }

  function clearRecipes() {
    activeRecipes = new Set();
  }

  function toggleStatus(status) {
    const next = new Set(activeStatuses);
    if (next.has(status)) next.delete(status);
    else next.add(status);
    activeStatuses = next;
  }

  function selectAllStatuses() {
    activeStatuses = new Set(statuses);
  }

  function clearStatuses() {
    activeStatuses = new Set();
  }

  function toggleGroup(group) {
    const next = new Set(activeGroups);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    activeGroups = next;
  }

  function selectAllGroups() {
    activeGroups = new Set(groups);
  }

  function clearGroups() {
    activeGroups = new Set();
  }

  function setActivePage(page) {
    activePage = page;
    activeTrainingRunId = null;
    drawerRunId = null;
    if (typeof window === "undefined") return;
    const targetPath = pagePaths[page] || pagePaths.training;
    if (window.location.pathname !== targetPath) {
      window.history.pushState({}, "", targetPath);
    }
  }

  function backToTrainingList() {
    activeTrainingRunId = null;
    drawerRunId = null;
    if (typeof window === "undefined") return;
    if (window.location.pathname !== pagePaths.training) {
      window.history.pushState({}, "", pagePaths.training);
    }
  }

  // Opening a run shows the full detail page (a real route, not a drawer).
  function openTrainingRunDetail(runId) {
    drawerRunId = null;
    activeTrainingRunId = runId;
    if (typeof window === "undefined") return;
    const target = `${pagePaths.training}/${encodeURIComponent(runId)}`;
    if (window.location.pathname !== target) {
      window.history.pushState({}, "", target);
    }
  }

  // "Collapse" on the detail page drops back to the list and reopens the run
  // as a summary drawer.
  function collapseTrainingRunToDrawer() {
    drawerRunId = activeTrainingRunId;
    activeTrainingRunId = null;
    if (typeof window === "undefined") return;
    if (window.location.pathname !== pagePaths.training) {
      window.history.pushState({}, "", pagePaths.training);
    }
  }

  function closeTrainingDrawer() {
    drawerRunId = null;
  }
</script>

<div class="h-[100dvh] grid grid-rows-[auto_1fr] bg-(--bg) overflow-x-hidden">
  <header class="[border-bottom:1px_solid_var(--color-c-surface-highlight-gray-opaque,#272727)] bg-(--bg-depth) flex items-center justify-between gap-[1rem] min-h-[53px] p-[0_1rem] max-[900px]:min-h-[53px] max-[900px]:p-[0_0.75rem]">
    <div class="inline-flex items-center gap-[0.55rem] flex-[0_0_auto] min-w-0">
      <img src={logoSvg} alt="Modal" class="h-[17.5px] w-auto flex-[0_0_auto]" />
      <span class="inline-flex items-center gap-[0.18rem] [font-family:var(--font-display)] [font-feature-settings:'ss01'_on] text-[17.6px] leading-[1] [padding-block:0.08rem] font-[600] tracking-[-0.02em] [transform:translateY(1px)] whitespace-nowrap max-[360px]:text-[15px]">
        <span class="text-[#ddffdc]">Modal</span>
        <span class="text-(--green)">Training Gym</span>
      </span>
    </div>
    <a
      class="[border:0] rounded-[10px] text-(--text) [background:transparent] [text-decoration:none] text-[14px] font-medium p-[8px] inline-flex items-center gap-[8px] flex-[0_0_auto] hover:text-(--text-bright) hover:[background:color-mix(in_srgb,white_4%,transparent)] max-[520px]:hidden"
      href={DOCS_URL}
      target="_blank"
      rel="noopener noreferrer"
    >
      <Book size={14} strokeWidth={2.1} />
      <span>Docs</span>
    </a>
  </header>

  <div class="grid grid-cols-[232px_minmax(0,1fr)] min-h-0 h-full bg-(--bg) max-[900px]:grid-cols-[1fr] max-[900px]:grid-rows-[auto_minmax(0,1fr)]">
    <Sidebar {navItems} {activePage} onNavigate={setActivePage} />

    <main class="min-w-0 min-h-0 h-full flex flex-col overflow-y-auto">
      <DashboardHeader
        title={pageMeta[activePage].title}
        {statusText}
        {refreshing}
        onRefresh={load}
      />

    {#if activePage === "training" && activeTrainingRunId}
      <TrainingRunDetailPage
        runId={activeTrainingRunId}
        initialRun={activeTrainingRun}
        {modelName}
        {getStatus}
        {getFrameworkStatus}
        {showFrameworkStatus}
        {fmtDuration}
        onBack={backToTrainingList}
        onCollapse={collapseTrainingRunToDrawer}
      />
    {:else if activePage === "training"}
      <TrainingPage
        {allRuns}
        {completedTotal}
        {runningTotal}
        {stoppedTotal}
        {failedTotal}
        {recipes}
        {recipeCounts}
        {activeRecipes}
        {statuses}
        {statusCounts}
        {activeStatuses}
        {groups}
        {groupCounts}
        {activeGroups}
        {filteredRuns}
        runGroups={trainingRunGroups}
        bind:groupBy={trainingGroupBy}
        {loading}
        {error}
        {modelName}
        {getStatus}
        {showFrameworkStatus}
        {fmtDuration}
        bind:search
        {drawerRunId}
        onOpenDetail={openTrainingRunDetail}
        onCloseDrawer={closeTrainingDrawer}
        onToggleRecipe={toggleRecipe}
        onSelectAllRecipes={selectAllRecipes}
        onClearRecipes={clearRecipes}
        onToggleStatus={toggleStatus}
        onSelectAllStatuses={selectAllStatuses}
        onClearStatuses={clearStatuses}
        onToggleGroup={toggleGroup}
        onSelectAllGroups={selectAllGroups}
        onClearGroups={clearGroups}
      />
    {:else if activePage === "evals"}
      <EvalsPage
        {allEvals}
        {evalCompletedTotal}
        {evalPendingTotal}
        {evalFailedTotal}
        loading={loadingEvals}
        {error}
        {evalConfigGroups}
        {fetchEvalDetail}
        {getEvalDisplay}
        {evalConfigMeta}
      />
    {/if}
    </main>
  </div>
</div>
