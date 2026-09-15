<!--
  A Balatro-specific trajectory viewer that renders game-state checkpoints,
  actions, cards, jokers, consumables, and vouchers.

  This is an example of a run-scoped DashboardComponent.TRAJECTORY_VIEWER.
  Attach it with:

    TrainingRun.from_id(...).add_dashboard_component(
        name="balatro",
        component_type=DashboardComponent.TRAJECTORY_VIEWER,
        from_path="examples/dashboard_components/BalatroTrajectoryViewer.svelte",
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
  let response = $derived(sample?.response || "");

  function parseStructuredMessage(msg) {
    const role = String(msg.role || "").toLowerCase();
    let raw = "";
    if (typeof msg.content === "string") {
      raw = msg.content;
    } else if (Array.isArray(msg.content)) {
      raw = msg.content
        .map((part) => (typeof part === "string" ? part : part?.text || ""))
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

  function parseToolCallBlock(block) {
    try {
      const data = JSON.parse(block);
      if (data.name) return { name: data.name, arguments: data.arguments || {} };
    } catch {}

    const functionMatch = block.match(
      /<function=([^>\n]+)>\s*([\s\S]*?)(?:<\/function>|$)/,
    );
    if (functionMatch) {
      const name = functionMatch[1].trim();
      const parameterRegex =
        /<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g;
      const args = {};
      let parameterMatch;
      while ((parameterMatch = parameterRegex.exec(functionMatch[2])) !== null) {
        try {
          args[parameterMatch[1].trim()] = JSON.parse(parameterMatch[2]);
        } catch {
          args[parameterMatch[1].trim()] = parameterMatch[2].trim();
        }
      }
      return { name, arguments: args };
    }

    const argKeyIndex = block.indexOf("<arg_key>");
    if (argKeyIndex >= 0) {
      const name = block.slice(0, argKeyIndex).trim();
      const argumentRegex =
        /<arg_key>\s*([\s\S]*?)\s*<\/arg_key>\s*<arg_value>\s*([\s\S]*?)\s*<\/arg_value>/g;
      const args = {};
      let argumentMatch;
      while ((argumentMatch = argumentRegex.exec(block)) !== null) {
        try {
          args[argumentMatch[1].trim()] = JSON.parse(argumentMatch[2]);
        } catch {
          args[argumentMatch[1].trim()] = argumentMatch[2].trim();
        }
      }
      if (name) return { name, arguments: args };
    }

    return null;
  }

  function formatArgs(args) {
    if (!args || !Object.keys(args).length) return "";
    try {
      return JSON.stringify(args, null, 2);
    } catch {
      return String(args);
    }
  }

  function jsonObjectEnd(text, start) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let index = start; index < text.length; index += 1) {
      const char = text[index];
      if (inString) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === '"') inString = false;
        continue;
      }
      if (char === '"') inString = true;
      else if (char === "{") depth += 1;
      else if (char === "}") {
        depth -= 1;
        if (depth === 0) return index + 1;
      }
    }
    return -1;
  }

  function parseBalatroResponse(text) {
    if (typeof text !== "string" || !text.includes("<balatro_state>")) return [];
    const turns = [];
    let cursor = 0;
    while (cursor < text.length) {
      const actionMatch = /\{"method"\s*:/.exec(text.slice(cursor));
      if (!actionMatch) break;
      const actionStart = cursor + actionMatch.index;
      const actionEnd = jsonObjectEnd(text, actionStart);
      if (actionEnd < 0) break;
      const stateStart = text.indexOf("<balatro_state>", actionEnd);
      if (stateStart < 0) break;
      const stateClose = "</balatro_state>";
      const stateEnd = text.indexOf(stateClose, stateStart);
      if (stateEnd < 0) break;
      const nextAction = text.indexOf('{"method"', stateEnd + stateClose.length);
      const toolEnd = nextAction >= 0 ? nextAction : text.length;
      turns.push({
        role: "assistant",
        thinking: null,
        content: text.slice(actionStart, actionEnd).trim(),
        toolCalls: [],
      });
      turns.push({
        role: "tool",
        thinking: null,
        content: text.slice(stateStart, toolEnd).trim(),
        toolCalls: [],
      });
      cursor = toolEnd;
    }
    return turns;
  }

  function extractGameStates(content) {
    const snapshots = [];
    const text = typeof content === "string" ? content : "";
    const pattern = /<balatro_state>([\s\S]*?)<\/balatro_state>/g;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      try {
        const payload = JSON.parse(match[1]);
        if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
          continue;
        }
        const state =
          payload.state && typeof payload.state === "object"
            ? payload.state
            : payload;
        snapshots.push({ payload, state });
      } catch {
        // Keep showing the raw tool result when a custom state is malformed.
      }
    }
    return snapshots;
  }

  function parseJsonAfter(text, marker) {
    const markerIndex = text.indexOf(marker);
    if (markerIndex < 0) return null;
    const start = text.indexOf("{", markerIndex + marker.length);
    if (start < 0) return null;
    try {
      return JSON.parse(text.slice(start));
    } catch {
      try {
        return JSON.parse(text.slice(start).match(/^\{[\s\S]*\}/)?.[0] || "");
      } catch {
        return null;
      }
    }
  }

  function actionFromMessage(message) {
    if (!message || message.role !== "assistant") return null;
    const structured = message.toolCalls?.at(-1);
    if (structured) {
      return { method: structured.name, params: structured.arguments || {} };
    }
    const text = typeof message.content === "string" ? message.content : "";
    const match = text.match(/\{[\s\S]*?"method"\s*:\s*"[^"]+"[\s\S]*\}/);
    if (!match) return null;
    try {
      const action = JSON.parse(match[0]);
      return action?.method ? action : null;
    } catch {
      return null;
    }
  }

  function actionBefore(index) {
    for (let indexBefore = index - 1; indexBefore >= 0; indexBefore -= 1) {
      const action = actionFromMessage(parsed[indexBefore]);
      if (action) return action;
    }
    return null;
  }

  function stateBefore(index) {
    for (let indexBefore = index - 1; indexBefore >= 0; indexBefore -= 1) {
      const message = parsed[indexBefore];
      const snapshots = extractGameStates(message?.content);
      if (snapshots.length) return snapshots.at(-1).state;
      const initial = parseJsonAfter(
        typeof message?.content === "string" ? message.content : "",
        "Gamestate:",
      );
      if (initial && typeof initial === "object" && !Array.isArray(initial)) {
        return initial;
      }
    }
    return null;
  }

  function stateCard(card) {
    const value =
      typeof card === "string"
        ? card
        : card?.id || card?.key || card?.label || "?";
    const match = String(value).match(/^([SHDC])_(.+)$/);
    if (!match) return { rank: String(value), suit: "", red: false };
    const suits = { S: "♠", H: "♥", D: "♦", C: "♣" };
    return {
      rank: match[2] === "T" ? "10" : match[2],
      suit: suits[match[1]],
      red: match[1] === "H" || match[1] === "D",
    };
  }

  function stateArea(state, name) {
    const area = state?.[name];
    if (Array.isArray(area)) return area;
    if (Array.isArray(area?.items)) return area.items;
    if (Array.isArray(area?.cards)) return area.cards;
    return [];
  }

  function itemId(item) {
    if (typeof item === "string") return item;
    return item?.id || item?.key || item?.label || "?";
  }

  function itemName(item) {
    const id = String(itemId(item));
    const name = id.replace(/^j_/, "").replace(/^c_/, "").replace(/^v_/, "");
    return (
      name
        .split(/[_-]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ") || id
    );
  }

  function itemEffect(item) {
    return item?.effect || item?.description || "";
  }

  function purchasedShopItem(action, priorState) {
    if (action?.method !== "buy") return null;
    const index = action.params?.card;
    if (!Number.isInteger(index)) return null;
    return stateArea(priorState, "shop")[index] || null;
  }

  function stateStatus(snapshot) {
    if (snapshot.payload?.error) return "invalid action · fallback used";
    if (snapshot.payload?.fallback) return "fallback action";
    return "state checkpoint";
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

  let parsed = $derived.by(() => {
    if (Array.isArray(messages) && messages.length > 0) {
      const structured = messages.map(parseStructuredMessage);
      const reconstructed = parseBalatroResponse(response);
      const actionTurns = structured.filter(
        (turn) => turn.role === "assistant",
      ).length;
      const lastRole = structured[structured.length - 1]?.role;
      const covered = actionTurns * 2 - (lastRole === "assistant" ? 1 : 0);
      return reconstructed.length > covered
        ? structured.concat(reconstructed.slice(covered))
        : structured;
    }
    if (response) {
      return [parseStructuredMessage({ role: "assistant", content: response })];
    }
    return [];
  });

  let thinkingOpen = $state({});
  function toggleThinking(index) {
    thinkingOpen = { ...thinkingOpen, [index]: !thinkingOpen[index] };
  }
</script>

<div class="trajectory-viewer">
  {#if parsed.length}
    {#each parsed as msg, index (index)}
      <div class:turn-tool={msg.role === "tool"} class="turn turn-{msg.role}">
        <div class="turn-header">
          <span class="role-label role-{msg.role}">{roleLabel(msg.role)}</span>
          {#if msg.role === "assistant" && index > 0}
            <span class="turn-meta">Turn {Math.ceil((index + 1) / 2)}</span>
          {/if}
          {#if msg.role === "assistant" && msg.toolCalls.length}
            <span class="turn-meta">
              {msg.toolCalls.length} tool call{msg.toolCalls.length > 1 ? "s" : ""}
            </span>
          {/if}
        </div>

        {#if msg.thinking}
          <button
            class="thinking-toggle"
            type="button"
            onclick={() => toggleThinking(index)}
          >
            <span class:open={thinkingOpen[index]}>⌄</span>
            <span>Thinking</span>
          </button>
          {#if thinkingOpen[index]}
            <pre class="thinking-block">{msg.thinking}</pre>
          {/if}
        {/if}

        {#if msg.content}
          {#if msg.role === "tool"}
            <details class="tool-raw-output">
              <summary>raw tool result</summary>
              <pre>{msg.content}</pre>
            </details>
          {:else}
            <pre class="message-content">{msg.content}</pre>
          {/if}
        {/if}

        {#if msg.role === "tool"}
          {@const gameStates = extractGameStates(msg.content)}
          {@const action = actionBefore(index)}
          {@const priorState = stateBefore(index)}
          {#if action}
            {@const actionCards = Array.isArray(action.params?.cards)
              ? action.params.cards
              : []}
            {@const purchased = purchasedShopItem(action, priorState)}
            <div
              class:play-action={action.method === "play"}
              class:discard-action={action.method === "discard"}
              class="tool-action-highlight"
            >
              <span class="tool-action-method">{action.method}</span>
              {#if actionCards.length}
                <span
                  >{actionCards.length} card{actionCards.length === 1 ? "" : "s"}
                  selected</span
                >
                <span class="tool-action-indices"
                  >[{actionCards.join(", ")}]</span
                >
              {/if}
            </div>
            {#if purchased && String(itemId(purchased)).startsWith("j_")}
              <div class="joker-purchase-callout">
                <span class="joker-purchase-glyph">★</span>
                <span><b>joker bought</b> {itemName(purchased)}</span>
                {#if purchased.effect}<small>{purchased.effect}</small>{/if}
              </div>
            {/if}
            {#if actionCards.length &&
              (action.method === "play" || action.method === "discard")}
              {@const priorHand = stateArea(priorState, "hand")}
              {#if priorHand.length}
                <div class="tool-action-table">
                  <div class="tool-action-table-title">
                    <span>before {action.method}</span>
                    <small>highlighted cards selected</small>
                  </div>
                  <div class="tool-action-hand full-hand">
                    {#each priorHand as selectedCard, cardIndex (`${index}-${cardIndex}-${JSON.stringify(selectedCard)}`)}
                      {@const selectedDisplay = stateCard(selectedCard)}
                      {@const isSelected = actionCards.includes(cardIndex)}
                      <span
                        class:selected-card={isSelected}
                        class:unselected-card={!isSelected}
                        class="action-card-wrap"
                      >
                        <span
                          class:red={selectedDisplay.red}
                          class:action-selected={isSelected}
                          class="tool-playing-card action-playing-card"
                          ><b>{selectedDisplay.rank}</b><i
                            >{selectedDisplay.suit}</i
                          ></span
                        >
                        <span class="action-card-index"
                          >{selectedDisplay.rank}{selectedDisplay.suit} ·
                          {cardIndex}</span
                        >
                      </span>
                    {/each}
                  </div>
                </div>
              {:else}
                <div class="tool-action-hand">
                  {#each actionCards as cardIndex, cardIndexPosition (`${index}-${cardIndexPosition}-${cardIndex}`)}
                    <span class="action-card-wrap">
                      <span class="tool-playing-card action-playing-card"
                        ><b>?</b><i></i
                      ></span>
                      <span class="action-card-index">card {cardIndex}</span>
                    </span>
                  {/each}
                </div>
              {/if}
            {/if}
          {/if}
          {#if gameStates.length}
            <div class="tool-state-viewer">
              {#each gameStates as snapshot, stateIndex (stateIndex)}
                {@const gameState = snapshot.state}
                {@const stateHand = stateArea(gameState, "hand")}
                {@const stateDeck = stateArea(gameState, "deck")}
                {@const stateDiscard = stateArea(gameState, "discard")}
                {@const stateJokers = stateArea(gameState, "jokers")}
                {@const stateConsumables = stateArea(gameState, "consumables")}
                {@const stateVouchers = stateArea(gameState, "vouchers")}
                {@const stateOwnedVouchers = stateArea(gameState, "owned_vouchers")}
                {@const purchased = purchasedShopItem(action, priorState)}
                <details
                  class="tool-state-card"
                  open={stateIndex === gameStates.length - 1}
                >
                  <summary>
                    <span class="tool-state-dot"></span>
                    <span
                      >Game state{gameStates.length > 1
                        ? ` ${stateIndex + 1}`
                        : ""}</span
                    >
                    <span class="tool-state-summary"
                      >A{gameState.ante ?? "—"} · R{gameState.round ?? "—"} ·
                      {gameState.phase || gameState.state || "checkpoint"}</span
                    >
                  </summary>
                  <div class="tool-state-status">{stateStatus(snapshot)}</div>
                  <div class="tool-state-stats">
                    <span
                      ><b>blind</b>{gameState.blind?.type ||
                        gameState.blind?.name ||
                        "—"}</span
                    >
                    <span
                      ><b>score</b>{gameState.blind?.score?.[0] ??
                        gameState.blind?.score ??
                        "—"}{gameState.blind?.score?.[1] != null
                        ? ` / ${gameState.blind.score[1]}`
                        : ""}</span
                    >
                    <span><b>hands</b>{gameState.blind?.hands ?? "—"}</span>
                    <span
                      ><b>deck</b>{(gameState.deck_left ?? stateDeck.length) ||
                        "—"}</span
                    >
                    <span><b>discard</b>{stateDiscard.length}</span>
                    <span
                      ><b>money</b>{gameState.money != null
                        ? `$${gameState.money}`
                        : "—"}</span
                    >
                  </div>
                  {#if stateJokers.length || gameState.jokers?.slots}
                    <div class="tool-state-hand-label joker-label">
                      <span>jokers</span
                      ><small
                        >{gameState.jokers?.slots ||
                          `${stateJokers.length}/5`}</small
                      >
                    </div>
                    {#if stateJokers.length}
                      <div class="tool-state-jokers">
                        {#each stateJokers as joker, jokerIndex (`${stateIndex}-${jokerIndex}-${JSON.stringify(joker)}`)}
                          {@const jokerId = String(itemId(joker))}
                          {@const isPurchased =
                            action?.method === "buy" &&
                            purchased &&
                            jokerId === String(itemId(purchased))}
                          <div class:joker-purchased={isPurchased} class="tool-joker-card">
                            <div class="tool-joker-topline">
                              <span class="tool-joker-mark">★</span>
                              <span class="tool-joker-name">{itemName(joker)}</span>
                            </div>
                            {#if joker.effect}<div class="tool-joker-effect">
                              {joker.effect}
                            </div>{/if}
                            {#if joker.mod}<div class="tool-joker-mod">
                              {JSON.stringify(joker.mod)}
                            </div>{/if}
                            {#if joker.sell != null}<div class="tool-joker-sell">
                              sell ${joker.sell}
                            </div>{/if}
                          </div>
                        {/each}
                      </div>
                    {:else}
                      <div class="tool-state-empty">none owned</div>
                    {/if}
                  {/if}
                  {#if stateConsumables.length}
                    <div class="tool-state-hand-label loadout-label">
                      <span>consumables</span
                      ><small
                        >{gameState.consumables?.slots ||
                          `${stateConsumables.length}/2`}</small
                      >
                    </div>
                    <div class="tool-state-items consumable-items">
                      {#each stateConsumables as consumable, itemIndex (`${stateIndex}-consumable-${itemIndex}-${JSON.stringify(consumable)}`)}
                        <div class="tool-loadout-card consumable-card">
                          <div class="tool-loadout-topline">
                            <span class="tool-loadout-mark">✦</span
                            ><span class="tool-loadout-name"
                              >{itemName(consumable)}</span
                            >
                          </div>
                          {#if itemEffect(consumable)}<div class="tool-loadout-effect">
                            {itemEffect(consumable)}
                          </div>{/if}
                          {#if consumable.mod}<div class="tool-loadout-mod">
                            {JSON.stringify(consumable.mod)}
                          </div>{/if}
                          {#if consumable.sell != null}<div class="tool-loadout-value">
                            sell ${consumable.sell}
                          </div>{/if}
                        </div>
                      {/each}
                    </div>
                  {/if}
                  {#if stateVouchers.length}
                    <div class="tool-state-hand-label loadout-label">
                      <span>shop vouchers</span>
                    </div>
                    <div class="tool-state-items voucher-items">
                      {#each stateVouchers as voucher, itemIndex (`${stateIndex}-voucher-${itemIndex}-${JSON.stringify(voucher)}`)}
                        <div class="tool-loadout-card voucher-card">
                          <div class="tool-loadout-topline">
                            <span class="tool-loadout-mark">◆</span
                            ><span class="tool-loadout-name"
                              >{itemName(voucher)}</span
                            >
                          </div>
                          {#if itemEffect(voucher)}<div class="tool-loadout-effect">
                            {itemEffect(voucher)}
                          </div>{/if}
                          {#if voucher.cost != null}<div class="tool-loadout-value">
                            cost ${voucher.cost}
                          </div>{/if}
                        </div>
                      {/each}
                    </div>
                  {/if}
                  {#if stateOwnedVouchers.length}
                    <div class="tool-state-hand-label loadout-label">
                      <span>owned vouchers</span>
                    </div>
                    <div class="tool-state-items voucher-items">
                      {#each stateOwnedVouchers as voucher, itemIndex (`${stateIndex}-owned-voucher-${itemIndex}-${JSON.stringify(voucher)}`)}
                        <div class="tool-loadout-card voucher-card owned-voucher-card">
                          <div class="tool-loadout-topline">
                            <span class="tool-loadout-mark">◆</span
                            ><span class="tool-loadout-name"
                              >{itemName(voucher)}</span
                            >
                          </div>
                          {#if itemEffect(voucher)}<div class="tool-loadout-effect">
                            {itemEffect(voucher)}
                          </div>{/if}
                          <div class="tool-loadout-value">active</div>
                        </div>
                      {/each}
                    </div>
                  {/if}
                  {#if stateHand.length}
                    <div class="tool-state-hand-label">
                      {action ? "after action · " : "hand · "}{stateHand.length}
                    </div>
                    <div class="tool-state-hand">
                      {#each stateHand as card, cardIndex (`${stateIndex}-${cardIndex}-${JSON.stringify(card)}`)}
                        {@const cardDisplay = stateCard(card)}
                        <span
                          class:red={cardDisplay.red}
                          class="tool-playing-card"
                          ><b>{cardDisplay.rank}</b><i
                            >{cardDisplay.suit}</i
                          ></span
                        >
                      {/each}
                    </div>
                  {/if}
                </details>
              {/each}
            </div>
          {/if}
        {/if}

        {#if msg.toolCalls.length}
          <div class="tool-calls">
            {#each msg.toolCalls as call, callIndex (callIndex)}
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
  {:else}
    <div class="empty-state">No trajectory messages or response available.</div>
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
  .turn:last-child { border-bottom: 0; }
  .turn-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .role-label { padding: 2px 6px; border-radius: 3px; color: #d1d1d1; font-size: 10px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }
  .role-assistant { background: rgb(96 165 250 / 14%); color: #93c5fd; }
  .role-user { background: rgb(52 211 153 / 14%); color: #6ee7b7; }
  .role-tool { background: rgb(251 191 36 / 14%); color: #fbbf24; }
  .role-system, .role-developer { background: rgb(168 139 250 / 14%); color: #c4b5fd; }
  .turn-meta { color: #777; font-size: 10px; }
  .message-content, .thinking-block { max-height: 400px; margin: 0; overflow: auto; color: #d1d1d1; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .thinking-toggle { display: inline-flex; align-items: center; gap: 5px; margin-bottom: 6px; padding: 2px 8px; border: 1px solid #383838; border-radius: 4px; background: none; color: #999; cursor: pointer; font-size: 11px; }
  .thinking-toggle:hover { border-color: #555; color: #d1d1d1; }
  .thinking-toggle span:first-child { display: inline-block; font-size: 15px; line-height: 10px; transition: transform .15s ease; }
  .thinking-toggle span:first-child.open { transform: rotate(180deg); }
  .thinking-block { margin: 0 0 6px; padding: 7px 9px; border-left: 2px solid #8b5cf6; background: rgb(139 92 246 / 8%); color: #aaa; }
  .tool-calls { display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }
  .tool-call { padding: 6px 10px; border-left: 2px solid #fbbf24; border-radius: 0 4px 4px 0; background: rgb(251 191 36 / 5%); }
  .tool-call-name { margin-bottom: 2px; color: #fbbf24; font: 600 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
  .tool-call-args { max-height: 160px; margin: 0; overflow: auto; color: #d1d1d1; font: 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .empty-state { padding: 16px 0; color: #777; }
  .tool-state-viewer { margin-top: 8px; border: 1px solid #2e3b32; border-radius: 6px; background: #101813; }
  .tool-state-card + .tool-state-card { border-top: 1px solid #29342d; }
  .tool-state-card summary { display: flex; align-items: center; gap: 7px; padding: 7px 9px; color: #d5e0d7; font-size: 10px; font-weight: 650; cursor: pointer; list-style: none; }
  .tool-state-card summary::-webkit-details-marker { display: none; }
  .tool-state-card summary:hover { background: #17221b; }
  .tool-state-dot { width: 6px; height: 6px; flex: 0 0 auto; border-radius: 50%; background: #dfaa4d; box-shadow: 0 0 0 3px rgb(223 170 77 / 12%); }
  .tool-state-summary { margin-left: auto; color: #81968a; font: 9px ui-monospace, monospace; text-transform: uppercase; }
  .tool-state-status { padding: 0 9px 6px 22px; color: #d7a44b; font-size: 9px; }
  .tool-state-stats { display: flex; flex-wrap: wrap; gap: 5px 13px; padding: 0 9px 8px 22px; color: #afc0b3; font: 9px ui-monospace, monospace; }
  .tool-state-stats span { display: inline-flex; gap: 4px; }
  .tool-state-stats b { color: #71867a; font-weight: 500; text-transform: uppercase; }
  .tool-state-hand-label { padding: 0 9px 5px 22px; color: #81968a; font-size: 9px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
  .joker-label { display: flex; align-items: center; gap: 7px; }
  .joker-label small { color: #bd9a50; font: 9px ui-monospace, monospace; letter-spacing: 0; }
  .tool-state-jokers { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 6px; padding: 0 9px 10px 22px; }
  .tool-joker-card { min-height: 58px; padding: 7px 8px 6px; border: 1px solid #72502a; border-radius: 6px; background: linear-gradient(145deg, #3b2418, #21150f 70%); color: #f4d18a; box-shadow: inset 0 0 0 1px rgb(255 220 130 / 8%), 1px 2px 4px rgb(0 0 0 / 25%); }
  .tool-joker-card.joker-purchased { border-color: #f3c65e; box-shadow: 0 0 0 2px rgb(243 198 94 / 22%), 0 0 14px rgb(243 198 94 / 16%), inset 0 0 0 1px rgb(255 220 130 / 12%); animation: joker-buy-glow 1.2s ease-in-out 2; }
  .tool-joker-topline { display: flex; align-items: center; gap: 5px; min-width: 0; }
  .tool-joker-mark, .joker-purchase-glyph { color: #f3c65e; text-shadow: 0 0 5px rgb(243 198 94 / 38%); }
  .tool-joker-name { overflow: hidden; color: #ffe1a1; font-size: 10px; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
  .tool-joker-effect { margin-top: 4px; color: #c9b58d; font-size: 9px; line-height: 1.28; }
  .tool-joker-mod { margin-top: 3px; color: #9dc6a0; font: 8px ui-monospace, monospace; overflow-wrap: anywhere; }
  .tool-joker-sell { margin-top: 4px; color: #8fa692; font: 8px ui-monospace, monospace; text-transform: uppercase; }
  .tool-state-empty { padding: 0 9px 10px 22px; color: #62796a; font-size: 9px; font-style: italic; }
  .loadout-label { display: flex; align-items: center; gap: 7px; }
  .loadout-label small { color: #9ca6c4; font: 9px ui-monospace, monospace; letter-spacing: 0; }
  .tool-state-items { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 6px; padding: 0 9px 10px 22px; }
  .tool-loadout-card { min-height: 52px; padding: 7px 8px 6px; border: 1px solid #45516e; border-radius: 6px; background: linear-gradient(145deg, #252b45, #171b2c 72%); color: #dce5ff; box-shadow: inset 0 0 0 1px rgb(193 208 255 / 7%); }
  .tool-loadout-card.voucher-card { border-color: #77602d; background: linear-gradient(145deg, #40351c, #211b10 72%); color: #f5df9f; }
  .tool-loadout-topline { display: flex; align-items: center; gap: 5px; min-width: 0; }
  .tool-loadout-mark { color: #b7c9ff; text-shadow: 0 0 5px rgb(183 201 255 / 35%); }
  .voucher-card .tool-loadout-mark { color: #f2c866; text-shadow: 0 0 5px rgb(242 200 102 / 35%); }
  .tool-loadout-name { overflow: hidden; color: #edf2ff; font-size: 10px; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
  .voucher-card .tool-loadout-name { color: #ffe7a9; }
  .tool-loadout-effect { margin-top: 4px; color: #b5c1df; font-size: 9px; line-height: 1.28; }
  .voucher-card .tool-loadout-effect { color: #cdbf98; }
  .tool-loadout-mod { margin-top: 3px; color: #9dc6a0; font: 8px ui-monospace, monospace; overflow-wrap: anywhere; }
  .tool-loadout-value { margin-top: 4px; color: #98a9d0; font: 8px ui-monospace, monospace; text-transform: uppercase; }
  .voucher-card .tool-loadout-value { color: #bda869; }
  .joker-purchase-callout { display: flex; align-items: baseline; gap: 6px; margin-top: 5px; padding: 6px 9px; border: 1px solid #75552a; border-radius: 5px; background: linear-gradient(90deg, #21170e, #18130d); color: #d7c08a; font-size: 10px; }
  .joker-purchase-callout b { color: #f3c65e; font-weight: 750; text-transform: uppercase; }
  .joker-purchase-callout small { margin-left: auto; max-width: 52%; overflow: hidden; color: #a99878; font-size: 9px; text-overflow: ellipsis; white-space: nowrap; }
  @keyframes joker-buy-glow { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-2px); } }
  .tool-state-hand { display: flex; flex-wrap: wrap; gap: 5px; padding: 0 9px 10px 22px; }
  .tool-playing-card { display: grid; place-content: center; width: 34px; height: 45px; border: 1px solid #c9c6b9; border-radius: 4px; background: #f0eee5; color: #242b26; box-shadow: 1px 2px 3px rgb(0 0 0 / 28%); text-align: center; }
  .tool-playing-card b { font: 700 12px Georgia, serif; line-height: 1; }
  .tool-playing-card i { font-style: normal; font-size: 11px; line-height: 1; }
  .tool-playing-card.red { color: #a03e3b; }
  .tool-action-highlight { display: flex; align-items: center; gap: 7px; margin-top: 8px; padding: 6px 9px; border: 1px solid #3c3522; border-radius: 5px; background: #19160e; color: #b7ad92; font-size: 10px; }
  .tool-action-highlight.play-action { border-color: #6d5423; background: #211a0e; }
  .tool-action-highlight.discard-action { border-color: #553735; background: #1c1212; }
  .tool-action-method { color: #f2c866; font: 700 10px ui-monospace, SFMono-Regular, Menlo, monospace; text-transform: uppercase; }
  .discard-action .tool-action-method { color: #e58b83; }
  .tool-action-indices { margin-left: auto; color: #d8c58d; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; }
  .tool-action-table { margin: 7px 0 2px 12px; padding: 8px 10px 9px; border: 1px solid #315640; border-radius: 7px; background: radial-gradient(circle at 50% 0%, #1d523c, #123224 75%); box-shadow: inset 0 0 0 1px rgb(255 255 255 / 4%); }
  .tool-action-table-title { display: flex; align-items: baseline; gap: 8px; margin-bottom: 7px; color: #d6e7d9; font-size: 9px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
  .tool-action-table-title small { color: #98b29e; font-size: 8px; font-weight: 500; letter-spacing: .02em; text-transform: none; }
  .tool-action-hand { display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-end; margin: 6px 0 2px 12px; padding: 4px 0 2px; }
  .tool-action-table .tool-action-hand { margin: 0; justify-content: center; }
  .action-card-wrap { display: grid; justify-items: center; gap: 2px; }
  .action-playing-card { transition: opacity .15s ease, transform .15s ease, box-shadow .15s ease; }
  .action-selected { animation: play-card-pulse 1.15s ease-in-out infinite; border-color: #f2b84b; box-shadow: 0 0 0 2px rgb(242 184 75 / 22%), 0 5px 9px rgb(0 0 0 / 32%); transform: translateY(-3px); }
  .unselected-card { opacity: .52; }
  .unselected-card .action-playing-card { filter: saturate(.65); }
  .action-card-index { color: #c7b67f; font: 8px ui-monospace, SFMono-Regular, Menlo, monospace; }
  @keyframes play-card-pulse { 0%, 100% { box-shadow: 0 0 0 2px rgb(242 184 75 / 18%), 0 5px 9px rgb(0 0 0 / 30%); } 50% { box-shadow: 0 0 0 4px rgb(242 184 75 / 35%), 0 8px 14px rgb(0 0 0 / 38%); transform: translateY(-7px); } }
  @media (prefers-reduced-motion: reduce) { .action-selected { animation: none; } }
  .tool-raw-output { margin-top: 6px; color: #718278; font-size: 9px; }
  .tool-raw-output summary { cursor: pointer; user-select: none; }
  .tool-raw-output pre { max-height: 260px; margin: 5px 0 0; overflow: auto; color: #9a9a9a; font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
