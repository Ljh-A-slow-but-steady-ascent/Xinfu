import assert from "node:assert/strict";
import test from "node:test";
import { browserStorage } from "./storage.js";

function replaceStorage(t, descriptor) {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { configurable: true, ...descriptor });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else delete globalThis.localStorage;
  });
}

test("a browser that blocks storage can still keep and clear the current login", (t) => {
  replaceStorage(t, { get() { throw new DOMException("Blocked", "SecurityError"); } });
  assert.equal(browserStorage.getItem("blocked-login"), null);
  browserStorage.setItem("blocked-login", "session-token");
  assert.equal(browserStorage.getItem("blocked-login"), "session-token");
  browserStorage.removeItem("blocked-login");
  assert.equal(browserStorage.getItem("blocked-login"), null);
});

test("full browser storage does not prevent saving a working session", (t) => {
  replaceStorage(t, { value: {
    getItem: () => null,
    setItem() { throw new DOMException("Full", "QuotaExceededError"); },
    removeItem() {},
  } });
  browserStorage.setItem("full-storage-login", "current-session");
  assert.equal(browserStorage.getItem("full-storage-login"), "current-session");
  browserStorage.removeItem("full-storage-login");
  assert.equal(browserStorage.getItem("full-storage-login"), null);
});

test("existing persisted sessions remain readable", (t) => {
  replaceStorage(t, { value: { getItem: (key) => key === "saved-login" ? "existing-session" : null } });
  assert.equal(browserStorage.getItem("saved-login"), "existing-session");
});
