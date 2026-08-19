export const MAX_TERMINAL_TIMING_FAILURES = 3;
const TERMINAL_TIMING_MIN_SETTLE_WINDOW_MS = 30 * 1000;
export const TERMINAL_TIMING_SETTLE_WINDOW_MS = 2 * 60 * 1000;
const TERMINAL_TIMING_STABLE_READS = 2;

export function timingReadFingerprint(timings) {
  return JSON.stringify(timings ?? {});
}

export function terminalTimingReadSettled({
  previousFingerprint,
  fingerprint,
  identicalReads = 0,
  startedAt,
  now = Date.now(),
}) {
  const nextIdenticalReads =
    previousFingerprint === fingerprint ? identicalReads + 1 : 1;
  const elapsed = startedAt == null ? 0 : Math.max(0, now - startedAt);
  return {
    identicalReads: nextIdenticalReads,
    settled:
      (elapsed >= TERMINAL_TIMING_MIN_SETTLE_WINDOW_MS &&
        nextIdenticalReads >= TERMINAL_TIMING_STABLE_READS) ||
      elapsed >= TERMINAL_TIMING_SETTLE_WINDOW_MS,
  };
}

// A terminal run gets its own retry budget: failures seen while it was still
// running must not spend it, or a run that finishes right after a dashboard
// restart shows no timeline at all.
export function updateTimingFailureState(
  state,
  { terminal, success = false, stale = false },
) {
  if (success) {
    return {
      runningFailures: stale ? state.runningFailures : 0,
      terminalFailures: stale ? state.terminalFailures : 0,
      staleFailures: stale ? (terminal ? state.staleFailures + 1 : 0) : 0,
    };
  }
  if (terminal) {
    return { ...state, terminalFailures: state.terminalFailures + 1 };
  }
  return { ...state, runningFailures: state.runningFailures + 1, terminalFailures: 0 };
}

export function shouldRetryTerminalTiming(state) {
  return state.terminalFailures < MAX_TERMINAL_TIMING_FAILURES;
}
