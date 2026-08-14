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
  ...[
    ["run-header", "Run header"],
    ["run-summary-card", "Run summary card"],
    ["framework-stage-progress", "Framework stage progress"],
    ["reward-chart", "Reward chart"],
    ["custom-tag-charts", "Custom tag charts"],
    ["score-distribution", "Score distribution"],
    ["advantage-distribution", "Advantage distribution"],
    ["step-timings", "Step timings"],
    ["rollout-explorer", "Rollout explorer"],
    ["run-logs", "Run logs"],
  ].map(([id, title]) => ({
    id,
    scope: "builtin",
    title,
    component: id,
    code: null,
  })),
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
