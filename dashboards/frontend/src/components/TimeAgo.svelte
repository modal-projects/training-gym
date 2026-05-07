<script>
  import { formatDistance } from "date-fns";
  import { readable } from "svelte/store";
  import { fmtDate, toEpochSeconds } from "../lib/format.js";

  let {
    timestamp,
    updateInterval = 5000,
    allowFuture = false,
    falsyRepresentation = "",
    showJustNow = false,
  } = $props();

  let text = $state("");
  let normalizedSeconds = $derived(toEpochSeconds(timestamp));
  let hasTimestamp = $derived(normalizedSeconds != null && normalizedSeconds > 0);
  let datetime = $derived.by(() => {
    if (!hasTimestamp) return undefined;
    const date = new Date(normalizedSeconds * 1000);
    if (Number.isNaN(date.getTime())) return undefined;
    return date.toISOString();
  });

  const now = readable(Date.now(), (set) => {
    if (typeof window !== "undefined") {
      const id = window.setInterval(() => {
        set(Date.now());
      }, updateInterval);
      return () => window.clearInterval(id);
    }
  });

  $effect(() => {
    if (!hasTimestamp) {
      text = falsyRepresentation;
      return;
    }

    let valueMs = normalizedSeconds * 1000;
    if (!allowFuture && valueMs > $now) {
      valueMs = $now;
    }

    if (showJustNow && valueMs > $now - 60 * 1000) {
      text = "just now";
      return;
    }

    text = formatDistance(valueMs, $now, {
      addSuffix: true,
      includeSeconds: true,
    });
  });
</script>

<time
  {datetime}
  title={hasTimestamp ? fmtDate(normalizedSeconds) : undefined}
>
  {text}
</time>
