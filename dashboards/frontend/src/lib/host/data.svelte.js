import { fetchRun, fetchRuns } from "../api.js";

const SERVER = "/api";

function pointer(value, path) {
  if (!path) return value;
  if (!path.startsWith("/")) throw new Error("ptr must be an RFC 6901 JSON pointer");
  return path
    .slice(1)
    .split("/")
    .reduce((current, token) => {
      const key = token.replace(/~1/g, "/").replace(/~0/g, "~");
      return current?.[Array.isArray(current) ? Number(key) : key];
    }, value);
}

export function doc(address, { ptr = "", poll_ms: pollMs = 5000 } = {}) {
  let data = $state(null);
  let loading = $state(true);
  let error = $state(null);
  let timer;

  async function read() {
    loading = true;
    try {
      const response = await fetch(`${SERVER}/docs/${address}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = pointer(await response.json(), ptr);
      error = null;
    } catch (reason) {
      error = reason instanceof Error ? reason : new Error(String(reason));
    } finally {
      loading = false;
    }
  }

  read();
  if (typeof window !== "undefined" && pollMs > 0) {
    timer = window.setInterval(read, pollMs);
  }

  return {
    get data() {
      return data;
    },
    get loading() {
      return loading;
    },
    get error() {
      return error;
    },
    refresh: read,
    destroy() {
      if (timer) window.clearInterval(timer);
    },
  };
}

export const api = { fetchRun, fetchRuns };
