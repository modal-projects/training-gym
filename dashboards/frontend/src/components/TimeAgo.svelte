<script>
  import { formatDistance } from "date-fns";
  import { nowMs } from "../lib/clock.js";
  import { fmtDate, toEpochSeconds } from "../lib/format.js";

  let {
    timestamp,
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

  $effect(() => {
    if (!hasTimestamp) {
      text = falsyRepresentation;
      return;
    }

    let valueMs = normalizedSeconds * 1000;
    if (!allowFuture && valueMs > $nowMs) {
      valueMs = $nowMs;
    }

    if (showJustNow && valueMs > $nowMs - 60 * 1000) {
      text = "just now";
      return;
    }

    text = formatDistance(valueMs, $nowMs, {
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
