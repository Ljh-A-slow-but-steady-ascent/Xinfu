const API_BASE = import.meta.env?.VITE_API_URL || "";

function expireSession(status, path) {
  if (status === 401 && path !== "/v1/auth/login" && typeof window !== "undefined") {
    window.dispatchEvent(new Event("xinfu:unauthorized"));
  }
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function request(path, options = {}, token = "") {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!response.ok) {
    expireSession(response.status, path);
    let detail = "请求失败";
    const text = await response.text();
    try {
      const payload = JSON.parse(text);
      detail = payload.detail || detail;
    } catch {
      detail = text || detail;
    }
    if (Array.isArray(detail)) detail = detail.map((item) => item?.msg || item?.message || JSON.stringify(item)).join("；");
    else if (detail && typeof detail === "object") detail = detail.message || detail.error || JSON.stringify(detail);
    throw new ApiError(String(detail), response.status);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response;
}

export async function analyzeLesion(form, token, signal) {
  const job = await request("/v1/rare-disease/jobs", { method: "POST", body: form, signal }, token);
  const deadline = Date.now() + 20 * 60 * 1000;
  while (Date.now() < deadline) {
    if (signal?.aborted) throw new DOMException("已停止", "AbortError");
    const status = await request(`/v1/rare-disease/jobs/${job.job_id}`, { signal }, token);
    if (status.status === "COMPLETED") return status.result;
    if (status.status === "FAILED") throw new Error(status.error || "图片分析失败");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("图片分析等待超时，请稍后重试");
}

export const authApi = {
  login: (username, password) => request("/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  }),
  me: (token) => request("/v1/auth/me", {}, token),
  logout: (token) => request("/v1/auth/logout", { method: "POST" }, token),
};

export async function fetchBlob(path, token) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) { expireSession(response.status, path); throw new ApiError("资源读取失败", response.status); }
  return response.blob();
}
