<script>
  import { onMount } from "svelte";
  import { ArrowLeft, ExternalLink, Minimize2 } from "$host/icons";
  import { StatusPill } from "$host/components";
  import { fetchRun } from "$host/data";

  let {
    runId,
    initialRun = null,
    getStatus = (run) => run?.status || "pending",
    onBack,
    onCollapse,
  } = $props();

  let run = $state(null);
  let error = $state("");

  $effect(() => {
    if (initialRun?.run_id === runId) run = initialRun;
  });

  onMount(() => {
    let disposed = false;
    async function refresh() {
      try {
        const next = await fetchRun(runId);
        if (!disposed && next) {
          run = next;
          error = "";
        }
      } catch (reason) {
        if (!disposed && !run) error = reason?.message || String(reason);
      }
    }
    void refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  });

  let wandbLinks = $derived(
    run?.wandb_links?.length
      ? run.wandb_links
      : run?.config_summary?.wandb_url
        ? [{ label: "Open in W&B", url: run.config_summary.wandb_url }]
        : [],
  );
</script>

<section class="detail-header">
  <header class="flex flex-wrap items-center gap-x-[10px] gap-y-[8px] p-[0_24px] mb-[16px] max-[900px]:p-[0_16px]">
    <button
      type="button"
      class="inline-flex items-center gap-[6px] [background:none] [border:0] text-(--muted) cursor-pointer text-[13px] leading-[16px] min-h-[32px] p-[4px_8px] rounded-[6px] hover:text-(--text) hover:bg-(--color-c-gray-10,#2f2f2f) max-[900px]:basis-full"
      onclick={onBack}
    >
      <ArrowLeft size={14} strokeWidth={2.1} />
      <span>Back to runs</span>
    </button>
    {#if onCollapse}
      <button
        type="button"
        class="inline-flex items-center gap-[6px] [border:1px_solid_var(--border,#2f2f2f)] rounded-[6px] [background:none] text-(--muted) cursor-pointer [font:inherit] text-[12px] font-medium leading-[16px] min-h-[32px] p-[4px_8px] hover:text-(--text-bright) hover:border-(--border-strong,#4a4a4a)"
        onclick={onCollapse}
        title="Collapse to drawer"
      >
        <Minimize2 size={12} strokeWidth={2.1} />
        <span>Collapse</span>
      </button>
    {/if}
    {#each wandbLinks as link (link.url)}
      <a class="header-link wandb-link inline-flex items-center gap-[6px] min-h-[32px] leading-[16px]" href={link.url} target="_blank" rel="noopener noreferrer">
        <span>{link.label}</span>
        <ExternalLink size={12} strokeWidth={2.1} />
      </a>
    {/each}
    {#if run?.modal_app_url}
      <a class="header-link inline-flex items-center gap-[6px] min-h-[32px] leading-[16px]" href={run.modal_app_url} target="_blank" rel="noopener noreferrer">
        <span>Open in Modal</span>
        <ExternalLink size={12} strokeWidth={2.1} />
      </a>
    {/if}
  </header>

  {#if run}
    <div class="flex items-center gap-[16px] p-[0_24px] mb-[16px] max-[900px]:p-[0_16px] min-w-0">
      <h1 class="text-[22px] font-[600] text-(--text-bright) m-0 overflow-hidden text-ellipsis whitespace-nowrap min-w-0" title={run.run_id}>{run.run_id}</h1>
      <StatusPill status={getStatus(run)} />
    </div>
  {:else}
    <div class="detail-empty px-[24px]">{error || `Loading run ${runId}…`}</div>
  {/if}
</section>
