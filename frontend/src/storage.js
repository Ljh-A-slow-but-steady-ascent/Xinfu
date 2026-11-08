const memory = new Map();

// Some mobile browsers restrict storage; the current session can still run.
export const browserStorage = {
  getItem(key) {
    try {
      const value = globalThis.localStorage.getItem(key);
      if (value !== null) return value;
    } catch { /* Continue with the current in-memory session. */ }
    return memory.get(key) ?? null;
  },
  setItem(key, value) {
    memory.set(key, String(value));
    try { globalThis.localStorage.setItem(key, String(value)); } catch { /* Session stays in memory. */ }
  },
  removeItem(key) {
    memory.delete(key);
    try { globalThis.localStorage.removeItem(key); } catch { /* Session is already cleared. */ }
  },
};
