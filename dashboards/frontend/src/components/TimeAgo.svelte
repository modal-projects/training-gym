<script>
  import { formatDistance } from "date-fns";
  import { readable } from "svelte/store";
  import { fmtDate } from "../lib/format.js";

  let {
    timestamp,
    updateInterval = 5000,
    allowFuture = false,
    falsyRepresentation = "",
    showJustNow = false,
  } = $props();

  let text = $state("");

  const now = readable(Date.now(), (set) => {
    if (typeof window !== "undefined") {
      const id = window.setInterval(() => {
        set(Date.now());
      }, updateInterval);
      return () => window.clearInterval(id);
    }
  });

  $effect(() => {
    if (!timestamp) {
      text = falsyRepresentation;
      return;
    }

    let valueMs = Number(timestamp) * 1000;
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
  datetime={timestamp ? new Date(Number(timestamp) * 1000).toISOString() : undefined}
  title={timestamp ? fmtDate(timestamp) : undefined}
>
  {text}
</time>
