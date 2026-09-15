<!--
  A small generic trajectory viewer for ordinary chat and agent messages.

  This is an example of a run-scoped DashboardComponent.TRAJECTORY_VIEWER.
  Attach it with:

    TrainingRun.from_id(...).add_dashboard_component(
        name="conversation",
        component_type=DashboardComponent.TRAJECTORY_VIEWER,
        from_path="examples/dashboard_components/ConversationTrajectoryViewer.svelte",
    )
-->
<script>
  let {
    sample = null,
    samples = [],
    trajectory = [],
    rewardEvents = [],
    rollout = null,
    run = null,
    position = null,
  } = $props();

  let messages = $derived(
    Array.isArray(trajectory) && trajectory.length
      ? trajectory
      : sample?.metadata?.trajectory_messages ?? [],
  );

  function messageText(message) {
    if (typeof message?.content === "string") return message.content;
    if (Array.isArray(message?.content)) {
      return message.content
        .map((part) => (typeof part === "string" ? part : part?.text || ""))
        .filter(Boolean)
        .join("\n");
    }
    return "";
  }

  function parseMessage(message) {
    const role = String(message?.role || "assistant").toLowerCase();
    let content = messageText(message);
    let thinking = "";
    if (content.includes("</think>")) {
      const [head, tail] = content.split("</think>", 2);
      thinking = head.replace(/<think>/g, "").trim();
      content = tail.replace(/<think>/g, "").trim();
    } else {
      content = content.replace(/<think>/g, "").trim();
    }

    const toolCalls = [];
    const toolCallPattern = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
    let match;
    while ((match = toolCallPattern.exec(content)) !== null) {
      toolCalls.push(match[1].trim());
    }
    content = content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();
    return { role, thinking, content, toolCalls };
  }

  function roleLabel(role) {
    return (
      {
        assistant: "Assistant",
        user: "User",
        tool: "Tool Result",
        system: "System",
        developer: "Developer",
      }[role] || role
    );
  }

  function rewardText(event) {
    if (event == null) return "";
    if (typeof event === "string" || typeof event === "number") return String(event);
    if (event.name != null || event.value != null) {
      return `${event.name ?? "reward"}: ${typeof event.value === "object" ? JSON.stringify(event.value) : event.value ?? ""}`;
    }
    try {
      return JSON.stringify(event);
    } catch {
      return String(event);
    }
  }

  let parsed = $derived.by(() => {
    if (Array.isArray(messages) && messages.length) return messages.map(parseMessage);
    if (sample?.thinking || sample?.response) {
      return [
        {
          role: "assistant",
          thinking: sample.thinking || "",
          content: sample.response || "",
          toolCalls: [],
        },
      ];
    }
    return [];
  });
</script>

<div class="trajectory-viewer">
  {#each parsed as message, index (index)}
    <section class="turn">
      <div class="turn-header">
        <span class="role-label role-{message.role}">{roleLabel(message.role)}</span>
      </div>
      {#if message.thinking}
        <details class="thinking">
          <summary>Thinking</summary>
          <pre>{message.thinking}</pre>
        </details>
      {/if}
      {#if message.content}
        <pre class="content">{message.content}</pre>
      {/if}
      {#if message.toolCalls.length}
        <div class="tool-calls">
          {#each message.toolCalls as call, callIndex (callIndex)}
            <div class="tool-call">
              <div>Tool call</div>
              <pre>{call}</pre>
            </div>
          {/each}
        </div>
      {/if}
    </section>
  {/each}

  {#if rewardEvents?.length}
    <section class="rewards">
      <div class="reward-title">Rewards</div>
      {#each rewardEvents as event, index (index)}
        <div class="reward">{rewardText(event)}</div>
      {/each}
    </section>
  {/if}
</div>

<style>
  :global(body) {
    margin: 0;
    background: transparent;
    color: #d1d1d1;
    font: 12px/1.5 ui-sans-serif, system-ui, sans-serif;
  }
  .trajectory-viewer { display: flex; flex-direction: column; gap: 2px; }
  .turn { padding: 8px 0; border-bottom: 1px solid #292929; }
  .turn-header { margin-bottom: 6px; }
  .role-label { display: inline-block; padding: 2px 6px; border-radius: 3px; color: #d1d1d1; font-size: 10px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }
  .role-assistant { background: rgb(96 165 250 / 14%); color: #93c5fd; }
  .role-user { background: rgb(52 211 153 / 14%); color: #6ee7b7; }
  .role-tool { background: rgb(251 191 36 / 14%); color: #fbbf24; }
  .role-system, .role-developer { background: rgb(168 139 250 / 14%); color: #c4b5fd; }
  .content, .thinking pre, .tool-call pre { margin: 0; color: #d1d1d1; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .thinking { margin-bottom: 6px; color: #999; }
  .thinking summary { cursor: pointer; }
  .thinking pre { margin-top: 5px; padding: 7px 9px; border-left: 2px solid #8b5cf6; background: rgb(139 92 246 / 8%); color: #aaa; }
  .tool-calls { display: flex; flex-direction: column; gap: 4px; margin-top: 7px; }
  .tool-call { padding: 6px 10px; border-left: 2px solid #fbbf24; border-radius: 0 4px 4px 0; background: rgb(251 191 36 / 5%); color: #fbbf24; }
  .tool-call pre { margin-top: 3px; color: #d1d1d1; }
  .rewards { margin-top: 8px; padding-top: 8px; border-top: 1px solid #383838; }
  .reward-title { margin-bottom: 4px; color: #d1d1d1; font-weight: 600; }
  .reward { color: #a7f3d0; font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
