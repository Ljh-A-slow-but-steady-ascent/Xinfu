import assert from "node:assert/strict";
import test from "node:test";
import { analyzeLesion, request } from "./api.js";

test("protected 401 expires the UI session; wrong login credentials do not", async (t) => {
  const events = [];
  t.mock.method(globalThis, "fetch", async () => new Response('{"detail":"请先登录"}', { status: 401 }));
  const previous = globalThis.window;
  globalThis.window = { dispatchEvent: (event) => events.push(event.type) };
  t.after(() => { if (previous === undefined) delete globalThis.window; else globalThis.window = previous; });
  await assert.rejects(request("/v1/agent/runs/dispatch"), { status: 401 });
  assert.deepEqual(events, ["xinfu:unauthorized"]);
  await assert.rejects(request("/v1/auth/login"), { status: 401 });
  assert.equal(events.length, 1);
});

test("non-JSON upstream errors retain the HTTP status", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("Bad Gateway", { status: 502 }));
  await assert.rejects(request("/health"), { status: 502, message: "Bad Gateway" });
});

test("image analysis submits a job and reads its completed result", async (t) => {
  const paths = [];
  t.mock.method(globalThis, "fetch", async (path) => {
    paths.push(path);
    return Response.json(paths.length === 1
      ? { job_id: "image-job", status: "RUNNING" }
      : { status: "COMPLETED", result: { package_id: "analysis-result" } });
  });
  const result = await analyzeLesion(new FormData(), "session");
  assert.equal(result.package_id, "analysis-result");
  assert.deepEqual(paths, ["/v1/rare-disease/jobs", "/v1/rare-disease/jobs/image-job"]);
});
