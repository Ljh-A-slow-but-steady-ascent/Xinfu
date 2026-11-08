export function workspaceResourceId(type, item) {
  return item[{ cases: "case_id", reports: "report_id", documents: "document_id" }[type]];
}

export function conversationHistory(messages) {
  return messages.filter((item) => !item.operationError && !(item.role === "assistant" && !item.run
    && /^(操作未完成|文档上传失败|确认操作失败)[：:]/.test(String(item.content || "")))).slice(-12)
    .map(({ role, content }) => ({ role, message: content }));
}
