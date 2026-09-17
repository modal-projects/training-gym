<script>
  // Port of the Modal dashboard's `MetricsRangeDropdown.svelte`. Emits
  // `onupdate({ live, start, end })` (epoch seconds) for a preset, or
  // `onupdate(null)` for "Entire run". `now` is the clock presets count back
  // from (epoch seconds); a finished run passes the moment it ended so
  // "Past 1 hour" is its final hour.
  import ChevronDown from "lucide-svelte/icons/chevron-down";
  import { onDestroy } from "svelte";

  import { useGlobalHotkey } from "../lib/hotkey.svelte.js";
  import { formatDurationShort, getTimezoneAbbr } from "../lib/timeRange.js";
  import Dropdown from "./Dropdown.svelte";
  import DropdownItem from "./DropdownItem.svelte";
  import LiveIndicator from "./LiveIndicator.svelte";

  let {
    live,
    start,
    end,
    duration,
    entireRun = false,
    now = undefined,
    liveLabel = "now",
    onupdate,
    open = $bindable(false),
  } = $props();

  const nowMs = () => (now !== undefined ? now * 1000 : Date.now());

  let interval;

  function formatDuration(duration) {
    return formatDurationShort(duration ?? 0);
  }

  onDestroy(() => {
    window.clearInterval(interval);
  });

  const rangeOptions = {
    "5m": { name: "Past 5 minutes", secondsAgo: 5 * 60 },
    "15m": { name: "Past 15 minutes", secondsAgo: 15 * 60 },
    "1h": { name: "Past 1 hour", secondsAgo: 60 * 60 },
    "4h": { name: "Past 4 hours", secondsAgo: 4 * 60 * 60 },
    "1d": { name: "Past 1 day", secondsAgo: 24 * 60 * 60 },
    "2d": { name: "Past 2 days", secondsAgo: 2 * 24 * 60 * 60 },
    "1w": { name: "Past 1 week", secondsAgo: 7 * 24 * 60 * 60 },
    "1mo": { name: "Past 1 month", secondsAgo: 30 * 24 * 60 * 60 },
  };

  const ENTIRE_RUN_KEY = "all";
  const elements = [ENTIRE_RUN_KEY, ...Object.keys(rangeOptions)];

  function setTimeRange(secondsAgo) {
    const to = nowMs();
    const from = to - secondsAgo * 1000;
    onupdate?.({
      live: true,
      start: from / 1000,
      end: to / 1000,
    });
  }

  function setEntireRun() {
    onupdate?.(null);
  }

  function formatDateRange(live, start, end, duration) {
    if (!start || !end || !duration) {
      return "";
    }

    const formatter = new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
      hour12: true,
    });

    if (live) {
      const from = new Date(nowMs() - duration * 1000);
      return `${formatter.format(from)} – ${liveLabel}`;
    }

    const from = new Date(start * 1000);
    const to = new Date(end * 1000);
    return `${formatter.format(from)} – ${formatter.format(to)}`;
  }

  let timezoneAbbr = $derived(getTimezoneAbbr());
  let formattedDateRange = $state("");
  $effect(() => {
    now;
    liveLabel;
    formattedDateRange = formatDateRange(live, start, end, duration);
  });
  let formattedDuration = $derived(formatDuration(duration));

  $effect(() => {
    if (live) {
      interval = window.setInterval(() => {
        formattedDateRange = formatDateRange(live, start, end, duration);
      }, 10000);
      return () => window.clearInterval(interval);
    }
  });

  let selectedKey = $derived(
    entireRun
      ? ENTIRE_RUN_KEY
      : live && rangeOptions[formattedDuration]
        ? formattedDuration
        : null,
  );

  useGlobalHotkey("r", setEntireRun);
</script>

{#if formattedDateRange}
  <Dropdown bind:open align="full" {elements}>
    {#snippet button()}
      <button type="button" class="range-trigger">
        <div class="range-trigger-main">
          <div class="range-chip">
            {#if live}
              <LiveIndicator widthPx={6} />
            {/if}
            {entireRun ? "run" : formattedDuration}
          </div>
          {formattedDateRange}
        </div>
        <div class="range-trigger-aside">
          {#if timezoneAbbr}
            <span class="text-(--muted)">{timezoneAbbr}</span>
          {/if}
          <ChevronDown size={20} class="text-(--muted)" />
        </div>
      </button>
    {/snippet}

    <DropdownItem selected={selectedKey === ENTIRE_RUN_KEY} itemIndex={0} onclick={setEntireRun}>
      <div class="range-option">
        <div class={["range-key", selectedKey === ENTIRE_RUN_KEY && "is-selected"]}>run</div>
        Entire run
      </div>
    </DropdownItem>

    {#each Object.entries(rangeOptions) as [k, v], i (k)}
      {@const isSelected = selectedKey === k}
      <DropdownItem
        selected={isSelected}
        itemIndex={i + 1}
        onclick={() => {
          setTimeRange(v.secondsAgo);
        }}
      >
        <div class="range-option">
          <div class={["range-key", isSelected && "is-selected"]}>{k}</div>
          {v.name}
        </div>
      </DropdownItem>
    {/each}

    <hr />

    <div class="range-hint">
      To select a custom time period, click and drag along any chart.
    </div>
  </Dropdown>
{/if}

<style>
  .range-trigger {
    display: inline-flex;
    width: 100%;
    min-width: min(390px, 100%);
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    border-radius: 6px;
    border: 1px solid var(--border-strong, #464646);
    background: var(--color-surface-primary, #1c1c1c);
    color: var(--text);
    padding: 6px;
    font: inherit;
    font-size: 14px;
    line-height: 20px;
    cursor: pointer;
    white-space: nowrap;
  }

  .range-trigger:hover {
    background: var(--color-surface-primary-hover, rgba(255, 255, 255, 0.05));
  }

  .range-trigger:focus-visible {
    outline: 2px solid var(--accent-border);
    outline-offset: 1px;
  }

  .range-trigger-main {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .range-trigger-aside {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .range-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    border-radius: 3px;
    background: var(--color-surface-secondary, #262626);
    padding: 0 8px;
    font-variant-numeric: tabular-nums;
  }

  .range-option {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .range-key {
    width: 40px;
    border-radius: 3px;
    text-align: center;
    background: var(--color-surface-secondary, #262626);
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .range-key.is-selected {
    background: var(--color-c-gray-20, #3a3a3a);
    color: var(--text-bright);
  }

  .range-hint {
    margin: 12px;
    font-size: 12px;
    color: var(--muted);
  }
</style>
