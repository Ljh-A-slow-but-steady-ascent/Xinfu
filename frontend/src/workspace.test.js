import assert from "node:assert/strict";
import test from "node:test";
import { conversationHistory, workspaceResourceId } from "./workspace.js";

test("linked reports and documents retain their own resource identifiers", () => {
  const report = { case_id: "CASE_1", report_id: "REPORT_1" };
  const document = { case_id: "CASE_1", document_id: "DOC_1" };
  assert.equal(workspaceResourceId("cases", { case_id: "CASE_1" }), "CASE_1");
  assert.equal(workspaceResourceId("reports", report), "REPORT_1");
  assert.equal(workspaceResourceId("documents", document), "DOC_1");
});

test("operational errors do not become conversational facts", () => {
  assert.deepEqual(conversationHistory([
    { role: "user", content: "现在什么天气？" },
    { role: "assistant", operationError: true, content: "操作未完成：登录已过期，请重新登录。" },
    { role: "assistant", content: "操作未完成：登录已过期，请重新登录。" },
    { role: "user", content: "北京" },
  ]), [{ role: "user", message: "现在什么天气？" }, { role: "user", message: "北京" }]);
});
