import RunHeader from "./builtin/RunHeader.svelte";
import RunSummaryView from "./builtin/RunSummaryView.svelte";
import RolloutExplorerView from "./builtin/RolloutExplorerView.svelte";
import RunLogsView from "./builtin/RunLogsView.svelte";
import FrameworkStageProgressView from "./builtin/FrameworkStageProgressView.svelte";
import RewardChartView from "./builtin/RewardChartView.svelte";
import CustomTagChartsView from "./builtin/CustomTagChartsView.svelte";
import ScoreDistributionView from "./builtin/ScoreDistributionView.svelte";
import AdvantageDistributionView from "./builtin/AdvantageDistributionView.svelte";
import StepTimingsView from "./builtin/StepTimingsView.svelte";
import JsonView from "./builtin/JsonView.svelte";
import trainingRunLayout from "./layouts/training-run.default.json";
import runHeaderSource from "./builtin/RunHeader.svelte?raw";
import runSummarySource from "./builtin/RunSummaryView.svelte?raw";
import frameworkStageSource from "./builtin/FrameworkStageProgressView.svelte?raw";
import rewardChartSource from "./builtin/RewardChartView.svelte?raw";
import customTagsSource from "./builtin/CustomTagChartsView.svelte?raw";
import scoreDistributionSource from "./builtin/ScoreDistributionView.svelte?raw";
import advantageDistributionSource from "./builtin/AdvantageDistributionView.svelte?raw";
import stepTimingsSource from "./builtin/StepTimingsView.svelte?raw";
import rolloutExplorerSource from "./builtin/RolloutExplorerView.svelte?raw";
import runLogsSource from "./builtin/RunLogsView.svelte?raw";

export const builtinViews = {
  "run-header": RunHeader,
  "run-summary-card": RunSummaryView,
  "framework-stage-progress": FrameworkStageProgressView,
  "reward-chart": RewardChartView,
  "custom-tag-charts": CustomTagChartsView,
  "score-distribution": ScoreDistributionView,
  "advantage-distribution": AdvantageDistributionView,
  "step-timings": StepTimingsView,
  "rollout-explorer": RolloutExplorerView,
  "run-logs": RunLogsView,
  json: JsonView,
};

export const builtinViewDocs = [
  {
    id: "json",
    scope: "builtin",
    title: "JSON",
    component: "json",
    code: null,
  },
  {
    id: "run-header",
    scope: "builtin",
    title: "Run header",
    component: "run-header",
    code: null,
    source: runHeaderSource,
  },
  {
    id: "run-summary-card",
    scope: "builtin",
    title: "Run summary card",
    component: "run-summary-card",
    code: null,
    source: runSummarySource,
  },
  {
    id: "framework-stage-progress",
    scope: "builtin",
    title: "Framework stage progress",
    component: "framework-stage-progress",
    code: null,
    source: frameworkStageSource,
  },
  {
    id: "reward-chart",
    scope: "builtin",
    title: "Reward chart",
    component: "reward-chart",
    code: null,
    source: rewardChartSource,
  },
  {
    id: "custom-tag-charts",
    scope: "builtin",
    title: "Custom tag charts",
    component: "custom-tag-charts",
    code: null,
    source: customTagsSource,
  },
  {
    id: "score-distribution",
    scope: "builtin",
    title: "Score distribution",
    component: "score-distribution",
    code: null,
    source: scoreDistributionSource,
  },
  {
    id: "advantage-distribution",
    scope: "builtin",
    title: "Advantage distribution",
    component: "advantage-distribution",
    code: null,
    source: advantageDistributionSource,
  },
  {
    id: "step-timings",
    scope: "builtin",
    title: "Step timings",
    component: "step-timings",
    code: null,
    source: stepTimingsSource,
  },
  {
    id: "rollout-explorer",
    scope: "builtin",
    title: "Rollout explorer",
    component: "rollout-explorer",
    code: null,
    source: rolloutExplorerSource,
  },
  {
    id: "run-logs",
    scope: "builtin",
    title: "Run logs",
    component: "run-logs",
    code: null,
    source: runLogsSource,
  },
];

export const builtinLayouts = {
  "training-run.default": trainingRunLayout,
};

export function resolveScopedDoc(docs, id, context = {}) {
  const scopes = ["builtin", "org", "user", "run"];
  let resolved = null;
  for (const scope of scopes) {
    let candidates = docs.filter((doc) => doc?.id === id && doc?.scope === scope);
    if (scope === "run" && context.run_id) {
      candidates = candidates.filter(
        (doc) => doc.run_id === context.run_id || doc.scope_id === context.run_id,
      );
    }
    if (candidates.length) resolved = candidates[candidates.length - 1];
  }
  return resolved;
}

export function resolveLayout(layouts, id, context = {}) {
  const builtin = builtinLayouts[id];
  return resolveScopedDoc(builtin ? [builtin, ...layouts] : layouts, id, context);
}

export function resolveView(views, id, context = {}) {
  const docs = [...builtinViewDocs, ...views];
  return resolveScopedDoc(docs, id, context);
}
