<script>
  import { ChevronDown } from "lucide-svelte";

  let { messages = null, response = "", thinking = "", evalReport = null } = $props();

  // ── Parse structured trajectory_messages into renderable turns ──────
  function parseStructuredMessage(msg) {
    const role = String(msg.role || "").toLowerCase();
    let raw = "";
    if (typeof msg.content === "string") {
      raw = msg.content;
    } else if (Array.isArray(msg.content)) {
      raw = msg.content
        .map((p) => (typeof p === "string" ? p : p?.text || ""))
        .filter(Boolean)
        .join("\n");
    }

    let thinkingText = null;
    let content = raw;
    let toolCalls = [];

    if (role === "assistant") {
      if (content.includes("</think>")) {
        const [head, tail] = content.split("</think>", 2);
        thinkingText = head.replace(/<think>/g, "").trim() || null;
        content = tail.replace(/<think>/g, "");
      } else {
        content = content.replace(/<think>/g, "");
      }

      const toolCallRegex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
      let match;
      while ((match = toolCallRegex.exec(content)) !== null) {
        const call = parseToolCallBlock(match[1]);
        if (call) toolCalls.push(call);
      }
      content = content.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();

      content = content
        .replace(/<\|im_start\|>assistant/g, "")
        .replace(/<\|im_end\|>/g, "")
        .replace(/<\|assistant\|>/g, "")
        .replace(/<\|endoftext\|>/g, "")
        .trim();
    }

    return { role, thinking: thinkingText, content, toolCalls };
  }

  // ── Parse a flattened multi-turn response into turns ────────────────
  // The response from generate.py concatenates turns with delimiters:
  //   {assistant}\n\nOBSERVATION:\n{tool}\n\n{assistant}\n\nOBSERVATION:\n{tool}...
  // Also handles SYSTEM: and DEVELOPER: labels.
  function parseFlatResponse(text) {
    if (!text) return [];
    const delimiter = /\n\n(OBSERVATION|SYSTEM|DEVELOPER):\n/;
    if (!delimiter.test(text)) return [];

    const turns = [];
    const parts = text.split(delimiter);

    // parts[0] = first assistant content
    // parts[1] = "OBSERVATION" (captured label)
    // parts[2] = tool content + maybe next assistant
    // parts[3] = "OBSERVATION"
    // parts[4] = tool content + maybe next assistant ...

    if (parts[0].trim()) {
      turns.push({ role: "assistant", thinking: null, content: parts[0].trim(), toolCalls: [] });
    }

    for (let i = 1; i < parts.length; i += 2) {
      const label = parts[i];
      const rawContent = (parts[i + 1] || "").trim();
      if (!rawContent) continue;

      const role = label === "OBSERVATION" ? "tool" : label.toLowerCase();

      // rawContent may contain tool output followed by \n\n then next assistant turn.
      // Split: find where tool output ends and next assistant content begins.
      // Heuristic: look for known XML end markers followed by \n\n then non-XML text.
      const { toolPart, assistantPart } = splitToolAndAssistant(rawContent);

      if (toolPart) {
        turns.push({ role, thinking: null, content: toolPart, toolCalls: [] });
      }
      if (assistantPart) {
        let aThinking = null;
        let aContent = assistantPart;
        if (aContent.includes("</think>")) {
          const [head, tail] = aContent.split("</think>", 2);
          aThinking = head.replace(/<think>/g, "").trim() || null;
          aContent = tail.trim();
        } else {
          aContent = aContent.replace(/<think>/g, "").trim();
        }
        turns.push({ role: "assistant", thinking: aThinking, content: aContent, toolCalls: [] });
      }
    }

    return turns.length > 1 ? turns : [];
  }

  function splitToolAndAssistant(text) {
    // Look for the boundary between tool output and next assistant turn.
    // Tool output from mini-swe ends with patterns like:
    //   </next_response_contract>\n\n
    //   </output>\n\n
    // The next assistant content follows after \n\n.
    const markers = [
      "</next_response_contract>",
      "</output_tail>",
      "</output_head>",
      "</output>",
      "</warning>",
    ];

    let bestIdx = -1;
    for (const marker of markers) {
      let searchFrom = 0;
      while (true) {
        const idx = text.indexOf(marker, searchFrom);
        if (idx < 0) break;
        const afterMarker = idx + marker.length;
        const nextDoubleNewline = text.indexOf("\n\n", afterMarker);
        if (nextDoubleNewline >= 0 && nextDoubleNewline <= afterMarker + 4) {
          const afterNewlines = nextDoubleNewline + 2;
          const remaining = text.slice(afterNewlines).trim();
          if (remaining && !remaining.startsWith("<")) {
            bestIdx = Math.max(bestIdx, nextDoubleNewline);
          }
        }
        searchFrom = afterMarker;
      }
    }

    if (bestIdx >= 0) {
      return {
        toolPart: text.slice(0, bestIdx).trim(),
        assistantPart: text.slice(bestIdx).trim(),
      };
    }

    // Split on </think> boundary: when OBSERVATION content includes a system
    // rejection followed by the model's next thinking + response (mini-swe
    // format), </think> marks where thinking ends and the assistant turn begins.
    // We keep the thinking text as a <think>...</think> wrapper in assistantPart
    // so the downstream parser can extract it into the thinking toggle.
    const thinkEndIdx = text.indexOf("</think>");
    if (thinkEndIdx >= 0) {
      const afterThink = text.slice(thinkEndIdx + 8).trim();
      if (afterThink) {
        const thinkingContent = text.slice(0, thinkEndIdx).trim();
        return {
          toolPart: null,
          assistantPart: thinkingContent + "\n</think>\n\n" + afterThink,
        };
      }
    }

    // Fallback: look for double-newline followed by common assistant patterns
    const assistantPatterns = [
      /\n\n(THOUGHT:)/,
      /\n\n(I )/,
      /\n\n(Let me )/,
      /\n\n(Now )/,
      /\n\n(The )/,
      /\n\n(Next,? )/,
      /\n\n(Good[.,! ])/,
      /\n\n(OK[.,! ])/,
    ];
    for (const pattern of assistantPatterns) {
      const match = pattern.exec(text);
      if (match && match.index > text.length * 0.3) {
        return {
          toolPart: text.slice(0, match.index).trim(),
          assistantPart: text.slice(match.index).trim(),
        };
      }
    }

    return { toolPart: text, assistantPart: null };
  }

  function parseToolCallBlock(block) {
    try {
      const data = JSON.parse(block);
      if (data.name) return { name: data.name, arguments: data.arguments || {} };
    } catch {}

    const fnMatch = block.match(/<function=([^>\n]+)>\s*([\s\S]*?)(?:<\/function>|$)/);
    if (fnMatch) {
      const name = fnMatch[1].trim();
      const paramRegex = /<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g;
      const args = {};
      let pm;
      while ((pm = paramRegex.exec(fnMatch[2])) !== null) {
        try { args[pm[1].trim()] = JSON.parse(pm[2]); }
        catch { args[pm[1].trim()] = pm[2].trim(); }
      }
      return { name, arguments: args };
    }

    const argKeyIdx = block.indexOf("<arg_key>");
    if (argKeyIdx >= 0) {
      const name = block.slice(0, argKeyIdx).trim();
      const argRegex = /<arg_key>\s*([\s\S]*?)\s*<\/arg_key>\s*<arg_value>\s*([\s\S]*?)\s*<\/arg_value>/g;
      const args = {};
      let am;
      while ((am = argRegex.exec(block)) !== null) {
        try { args[am[1].trim()] = JSON.parse(am[2]); }
        catch { args[am[1].trim()] = am[2].trim(); }
      }
      if (name) return { name, arguments: args };
    }

    return null;
  }

  function formatArgs(args) {
    if (!args || !Object.keys(args).length) return "";
    try { return JSON.stringify(args, null, 2); }
    catch { return String(args); }
  }

  function roleLabel(role) {
    return { assistant: "Assistant", user: "User", tool: "Tool Result", system: "System", developer: "Developer" }[role] || role;
  }

  // Resolve turns: prefer structured messages, then multi-turn response
  // parsing, and finally render a plain single-turn response as one assistant
  // turn.
  let parsed = $derived.by(() => {
    if (Array.isArray(messages) && messages.length > 0) {
      return messages.map(parseStructuredMessage);
    }
    const turns = parseFlatResponse(response);
    if (turns.length) return turns;
    if (response) {
      return [parseStructuredMessage({ role: "assistant", content: response })];
    }
    return [];
  });

  let thinkingOpen = $state({});
  function toggleThinking(idx) {
    thinkingOpen = { ...thinkingOpen, [idx]: !thinkingOpen[idx] };
  }

  let checks = $derived.by(() => {
    if (!evalReport || typeof evalReport !== "object") return null;
    const c = evalReport.checks;
    if (!c || typeof c !== "object" || !Object.keys(c).length) return null;
    return Object.entries(c).map(([name, detail]) => ({
      name,
      passed: !!detail.passed,
      status: detail.status || "",
      score: detail.score,
      errors: detail.errors || [],
    }));
  });

  let checkSummary = $derived(evalReport?.check_summary || null);
</script>

{#if parsed.length}
<div class="conversation">
  {#if thinking}
    <div class="turn turn-thinking">
      <div class="turn-header">
        <span class="role-badge role-thinking">Thinking</span>
      </div>
      <pre class="thinking-block">{thinking}</pre>
    </div>
  {/if}

  {#each parsed as msg, idx (idx)}
    <div class="turn turn-{msg.role}">
      <div class="turn-header">
        <span class="role-badge role-{msg.role}">{roleLabel(msg.role)}</span>
        {#if msg.role === "assistant" && idx > 0}
          <span class="turn-meta">Turn {Math.ceil((idx + 1) / 2)}</span>
        {/if}
        {#if msg.role === "assistant" && msg.toolCalls.length}
          <span class="turn-meta">{msg.toolCalls.length} tool call{msg.toolCalls.length > 1 ? "s" : ""}</span>
        {/if}
      </div>

      {#if msg.thinking}
        <button class="thinking-toggle" onclick={() => toggleThinking(idx)}>
          <ChevronDown size={12} style={thinkingOpen[idx] ? "transform:rotate(180deg)" : ""} />
          <span>Thinking</span>
        </button>
        {#if thinkingOpen[idx]}
          <pre class="thinking-block">{msg.thinking}</pre>
        {/if}
      {/if}

      {#if msg.content}
        <pre class="turn-content" class:tool-output={msg.role === "tool"}>{msg.content}</pre>
      {/if}

      {#if msg.toolCalls.length}
        <div class="tool-calls">
          {#each msg.toolCalls as call, ci (ci)}
            <div class="tool-call">
              <div class="tool-call-name">{call.name}()</div>
              {#if Object.keys(call.arguments || {}).length}
                <pre class="tool-call-args">{formatArgs(call.arguments)}</pre>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/each}

  {#if checks}
    <div class="checks-section">
      <div class="checks-header">
        Checks
        {#if checkSummary}
          <span class="checks-summary">
            <span class="check-pass-count">{checkSummary.passed} passed</span>
            {#if checkSummary.failed}
              <span class="check-fail-count">{checkSummary.failed} failed</span>
            {/if}
          </span>
        {/if}
      </div>
      <div class="checks-list">
        {#each checks as check (check.name)}
          <div class="check-row" class:check-passed={check.passed} class:check-failed={!check.passed}>
            <span class="check-icon">{check.passed ? "✓" : "✗"}</span>
            <span class="check-name">{check.name}</span>
            {#if check.score != null && typeof check.score === "number"}
              <span class="check-score">{check.score.toFixed(2)}</span>
            {/if}
          </div>
          {#if !check.passed && check.errors.length}
            <pre class="check-error">{check.errors.join("\n")}</pre>
          {/if}
        {/each}
      </div>
    </div>
  {/if}
</div>
{/if}

<style>
  .conversation {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .turn {
    padding: 10px 12px;
    border-radius: 6px;
    background: var(--color-c-gray-08, #1c1c1c);
  }

  .turn + .turn {
    margin-top: 1px;
  }

  .turn-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }

  .role-badge {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 6px;
    border-radius: 3px;
  }

  .role-assistant {
    background: rgba(96, 165, 250, 0.12);
    color: #60a5fa;
  }

  .role-user {
    background: rgba(74, 222, 128, 0.12);
    color: #4ade80;
  }

  .role-tool {
    background: rgba(251, 191, 36, 0.12);
    color: #fbbf24;
  }

  .role-system, .role-developer, .role-thinking {
    background: rgba(168, 139, 250, 0.12);
    color: #a78bfa;
  }

  .turn-meta {
    font-size: 11px;
    color: var(--muted);
  }

  .turn-content {
    margin: 0;
    padding: 0;
    background: none;
    font-size: 12px;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow: auto;
    line-height: 1.5;
  }

  .tool-output {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    color: var(--muted);
    background: rgba(0, 0, 0, 0.2);
    padding: 8px;
    border-radius: 4px;
    max-height: 200px;
  }

  .thinking-toggle {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: none;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11px;
    padding: 2px 8px;
    cursor: pointer;
    margin-bottom: 6px;
  }

  .thinking-toggle:hover {
    color: var(--text);
    border-color: var(--border-strong, #4a4a4a);
  }

  .thinking-block {
    margin: 0 0 8px;
    padding: 8px;
    background: rgba(168, 139, 250, 0.06);
    border: 1px solid rgba(168, 139, 250, 0.15);
    border-radius: 4px;
    font-size: 11px;
    color: var(--muted);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow: auto;
    line-height: 1.45;
  }

  .tool-calls {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 6px;
  }

  .tool-call {
    border-left: 2px solid #fbbf24;
    padding: 6px 10px;
    border-radius: 0 4px 4px 0;
    background: rgba(251, 191, 36, 0.05);
  }

  .tool-call-name {
    font-size: 12px;
    font-weight: 600;
    color: #fbbf24;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    margin-bottom: 2px;
  }

  .tool-call-args {
    margin: 0;
    padding: 0;
    font-size: 11px;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 160px;
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    line-height: 1.4;
  }

  .checks-section {
    margin-top: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    background: var(--color-c-gray-08, #1c1c1c);
    border: 1px solid var(--border, #2f2f2f);
  }

  .checks-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-bright);
    margin-bottom: 8px;
  }

  .checks-summary {
    display: inline-flex;
    gap: 8px;
    font-size: 11px;
    font-weight: 400;
  }

  .check-pass-count { color: #4ade80; }
  .check-fail-count { color: #f87171; }

  .checks-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .check-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    padding: 2px 0;
  }

  .check-icon {
    flex-shrink: 0;
    width: 14px;
    text-align: center;
    font-weight: 600;
    font-size: 11px;
  }

  .check-passed .check-icon { color: #4ade80; }
  .check-failed .check-icon { color: #f87171; }

  .check-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text);
  }

  .check-score {
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
  }

  .check-error {
    margin: 2px 0 4px 20px;
    padding: 6px 8px;
    background: rgba(248, 113, 113, 0.06);
    border-radius: 4px;
    font-size: 11px;
    color: var(--muted);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 120px;
    overflow: auto;
  }
</style>
