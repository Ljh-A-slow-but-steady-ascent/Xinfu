import React, { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Archive, ArrowUp, ArrowUpRight, BarChart3, Check, ChevronLeft, Copy, Download, Eye, BookOpen, BrainCircuit, FilePenLine, FileText, FileUp, ImagePlus, Layers3, LoaderCircle, LogOut, Menu, MessageSquare, PanelLeftClose, PanelLeftOpen, Pencil, Play, Plus, Search, Database, ScanSearch, ShieldCheck, Square, Stethoscope, Trash2, X } from "lucide-react";
import { analyzeLesion, authApi, fetchBlob, request } from "./api";
import Login from "./Login";
import { BrandMark } from "./Brand";
import ApprovalPreview from "./ApprovalPreview";
import { conversationHistory, workspaceResourceId } from "./workspace";
import { browserStorage } from "./storage";

const KnowledgeCenter = React.lazy(() => import("./KnowledgeCenter"));
const LazyMarkdown = React.lazy(() => import("./RichMarkdown"));
function Markdown(props) {
  return <React.Suspense fallback={<span className="content-loading">正在打开内容…</span>}><LazyMarkdown {...props} /></React.Suspense>;
}

const TOKEN_KEY = "xinfu_auth_token";
const LABELS = {
  nevus: "色素痣", melanoma: "黑色素瘤", "basal cell carcinoma": "基底细胞癌",
  "squamous cell carcinoma": "鳞状细胞癌", "actinic keratosis": "日光性角化病",
  "seborrheic keratosis": "脂溢性角化病", dermatofibroma: "皮肤纤维瘤",
};
const REPORT_STATUS = { draft: "草稿", final: "已定稿" };
const LEVELS = {
  RETAKE: "建议重新拍摄", LOCAL_REVIEW: "基层继续评估", REFERRAL: "建议专科转诊",
  PRIORITY_REFERRAL: "建议优先转诊",
};


const MODE_LABELS = { zero_shot: "图文比对", image_text: "图文分析", image_only: "图片分析", text_only: "文字分析" };
function friendlyError(error, fallback = "暂时无法完成，请稍后重试。") {
  if (error?.status === 401) return "登录已过期，请重新登录。";
  if (error?.status >= 500) return "服务暂时没有响应，请稍后重试。";
  const message = String(error?.message || "");
  if (!message || message.length > 180 || /Method Not Allowed|Not Found|Forbidden|Bad Request|Internal Server Error|Failed to fetch|NetworkError|Traceback|Exception|Error:|API_KEY|DERMAGUARD|SQLAlchemy|\/home\/|\/media\/|\/tmp\/|ModuleNotFound|RuntimeError|checkpoint|密钥|提供方|模型不可用|模型响应|未配置.*模型/i.test(message)) return fallback;
  return message;
}

const uid = () => (typeof crypto !== "undefined" && crypto.randomUUID?.()) || `thread-${Date.now()}-${Math.random()}`;
const now = () => new Date().toISOString();
const threadKey = (userId) => `xinfu_threads_v2_${userId}`;

function freshThread() {
  return { id: uid(), title: "新对话", caseId: null, intakeDraft: "", preparedIntake: null, intakeState: "IDLE", selectedRareEvidenceId: null, createdAt: now(), updatedAt: now(), messages: [] };
}

function loadThreads(userId) {
  try {
    const parsed = JSON.parse(browserStorage.getItem(threadKey(userId)) || "[]");
    if (Array.isArray(parsed) && parsed.length) return parsed;
  } catch { /* use a clean workspace */ }
  return [freshThread()];
}

const rareKey = (userId) => `xinfu_rare_packages_${userId}`;

function loadRarePackages(userId) {
  try {
    const parsed = JSON.parse(browserStorage.getItem(rareKey(userId)) || "[]");
    if (Array.isArray(parsed)) return parsed;
  } catch { /* use a clean store */ }
  return [];
}

function titleFrom(text) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean ? clean.slice(0, 24) : "图像分析";
}

function recoverIntakeNarrative(messages) {
  const commandOnly = /^(?:开始分析|开始|继续|继续分析|按上文|按照上文|按上文记忆|按照上文记忆|执行|重新提交|再次提交|确认)$/;
  return (messages || [])
    .filter((item) => item.role === "user")
    .map((item) => String(item.content || "").trim())
    .filter((content) => content && !content.startsWith("[已附加") && !commandOnly.test(content.replace(/[\s。，,！!？?]/g, "")))
    .slice(-12)
    .join("\n补充信息：")
    .slice(-6000);
}

function looksLikeClinicalIntakeDocument(content) {
  const text = String(content || "");
  const signals = [
    /年龄\s*[：:]/, /性别\s*[：:]/, /病灶部位\s*[：:]/,
    /病灶尺寸\s*[：:]/, /瘙痒\s*[：:]/, /疼痛\s*[：:]/,
    /出血\s*[：:]/, /近期增大\s*[：:]/, /近期变化\s*[：:]/,
    /皮肤癌病史\s*[：:]/,
  ];
  return signals.filter((pattern) => pattern.test(text)).length >= 3;
}

function formatTime(value) {
  const date = new Date(value);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function Sidebar({ open, threads, activeId, activeView, user, onToggle, onNew, onSelect, onRename, onDelete, onWorkspace, onLogout }) {
  const [query, setQuery] = useState("");
  const visible = threads.filter((item) => item.title.toLowerCase().includes(query.toLowerCase()));
  const navigation = [
    ["cases", Archive, "病例库"], ["reports", FileText, "报告库"], ["documents", FilePenLine, "文档库"],
    ["knowledge", Database, "资料与研究"], ["rare", ScanSearch, "病灶分析"],
  ];
  return <aside className={`sidebar ${open ? "open" : "closed"}`}>
    <div className="sidebar-head"><BrandMark size={34} tone="light" />{open && <div className="sidebar-brand"><strong>信肤</strong><span>工作台</span></div>}<button className="icon-button push" aria-label={open ? "收起侧栏" : "展开侧栏"} onClick={onToggle}>{open ? <PanelLeftClose /> : <PanelLeftOpen />}</button></div>
    <button className="new-chat" onClick={onNew} title="新对话"><Plus size={17} />{open && <><span>新对话</span><ArrowUpRight size={14} /></>}</button>
    {open && <div className="side-label">工作空间</div>}
    <nav className="side-nav" aria-label="工作空间">{navigation.map(([type, Icon, label]) => <button key={type} className={activeView === type ? "selected" : ""} aria-current={activeView === type ? "page" : undefined} title={label} onClick={() => onWorkspace(type)}><Icon size={17} />{open && <span>{label}</span>}</button>)}</nav>
    {open && <><div className="side-label side-label-history">最近对话 <span>{threads.length}</span></div>
      <div className="sidebar-search"><Search size={14} /><input aria-label="搜索对话" placeholder="搜索对话" value={query} onChange={(e) => setQuery(e.target.value)} /></div>
      <div className="thread-list">{visible.map((thread) => <div role="button" tabIndex={0} key={thread.id} className={`thread-item ${thread.id === activeId && !activeView ? "active" : ""}`} onClick={() => onSelect(thread.id)} onKeyDown={(event) => { if (event.key === "Enter") onSelect(thread.id); }}>
        <MessageSquare size={15} /><div className="thread-copy"><strong>{thread.title}</strong><span>{formatTime(thread.updatedAt)}</span></div>
        <div className="thread-actions"><button title="重命名" onClick={(e) => { e.stopPropagation(); onRename(thread.id); }}><Pencil size={13} /></button><button title="删除对话" onClick={(e) => { e.stopPropagation(); onDelete(thread.id); }}><Trash2 size={13} /></button></div>
      </div>)}{!visible.length && <p className="sidebar-no-results">没有找到相关对话</p>}</div></>}
    <div className="profile"><div className="profile-avatar">{(user.full_name || user.username || "医")[0]}</div>{open && <><div><strong>{user.full_name || user.username}</strong><span>个人工作空间</span></div><button title="退出登录" onClick={onLogout}><LogOut size={16} /></button></>}</div>
  </aside>;
}

function ExecutionCard({ run, onStop }) {
  const [expanded, setExpanded] = useState(false);
  const events = run?.events || [];
  const labels = { route_request: "整理问题", agent_context: "读取上下文", agent_planning: "安排下一步", search_evidence: "查找相关资料", search_guideline: "查找相关资料", read_evidence: "阅读资料", multimodal_analysis: "查看图片与病例", agent_synthesis: "整理结果", completed: "处理完成", task_plan: "更新工作步骤", open_source_agent: "准备处理", get_visual_evidence: "读取病例记录", get_triage_result: "查看评估结果", web_search: "搜索公开网页", read_webpage: "阅读网页原文", get_weather: "查询天气" };
  const visible = events.slice(-6);
  return <div className={`execution-card ${run?.running ? "is-running" : "is-complete"}`}>
    <div className="execution-title">{run?.running ? <span className="activity-pulse"><i /><i /><i /></span> : <span className="execution-check"><Check size={13} /></span>}<strong>{run?.running ? "正在处理" : "处理完成"}</strong><button className="execution-expand" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}>{expanded ? "收起步骤" : "查看步骤"}<ChevronLeft size={12} /></button>{run?.running && <button className="execution-stop" onClick={onStop} title="停止等待"><Square size={11} /></button>}</div>
    {(expanded || run?.running) && <div className="event-list">{visible.length ? visible.map((event, index) => <div key={`${event.sequence}-${index}`} className={index === visible.length - 1 && run?.running ? "current" : "done"}><i /><span>{labels[event.node] || (event.event_type === "error" ? "暂时无法继续，请重试" : "更新工作记录")}</span></div>) : <div className={run?.running ? "current" : "done"}><i /><span>{run?.running ? "正在整理问题" : "处理完成"}</span></div>}</div>}
  </div>;
}

function ApprovalCard({ action, onConfirm, onReject, busy }) {
  const copy = {
    CREATE_ANALYZE_CASE: ["创建病例并开始分析", "将当前资料保存为病例，并开始分析图片。"],
    UPDATE_CASE: ["修改病例记录", "保存以下修改，并更新分析结果。"],
    GENERATE_REPORT: ["生成报告", "根据当前病例生成报告草稿，保存到报告库。"],
    DELETE_REPORT: ["删除报告", "删除所选报告。"], DELETE_DOCUMENT: ["删除文档", "删除所选文档。"],
  }[action.kind] || [action.title, action.summary];
  return <div className={`approval-card ${action.destructive ? "destructive" : ""}`}><div className="approval-icon"><ShieldCheck /></div><div className="approval-body"><strong>{copy[0]}</strong><p>{copy[1]}</p><ApprovalPreview preview={action.preview} /><div className="approval-actions"><button onClick={onReject} disabled={busy}>取消</button><button className="approve" onClick={onConfirm} disabled={busy}>{busy && <LoaderCircle className="spin" size={14} />}{action.destructive ? "确认删除" : "确认执行"}</button></div></div></div>;
}

function ArtifactCard({ item, token, onOpen }) {
  async function download() {
    const blob = await fetchBlob(`/v1/agent/documents/${encodeURIComponent(item.document_id)}/download`, token);
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = item.filename || "agent-document.md"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 500);
  }
  return <div className="artifact-card"><div className="artifact-icon"><FilePenLine size={19} /></div><div><strong>{item.title}</strong><span>{item.filename} · 第 {item.revision || 1} 版</span></div><button onClick={() => onOpen({ type: "document", data: item })}>编辑</button><button onClick={download}><Download size={14} />下载</button></div>;
}

function Message({ message, token, onOpen, onApprove, approving, rareEvidencePackages, onOpenEvidence }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => { try { await navigator.clipboard.writeText(message.content || ""); setCopied(true); setTimeout(() => setCopied(false), 1600); } catch { setCopied(false); } };
  const rarePkg = message.rareEvidenceId ? (rareEvidencePackages || []).find((item) => item.package_id === message.rareEvidenceId) : null;
  return <article className={`message ${message.role}`}>
    {message.role === "assistant" && <div className="assistant-avatar"><BrandMark size={29} /></div>}
    <div className="message-column"><div className="message-content"><Markdown token={token} rarePackages={rareEvidencePackages}>{message.content}</Markdown></div>
      {rarePkg && <div className="message-evidence-card">
        <div className="message-evidence-head"><BrainCircuit size={14} /><strong>病灶分析 · {formatTime(rarePkg.created_at || now())}</strong><button onClick={() => onOpenEvidence?.(rarePkg)}><ArrowUpRight size={13} />展开详情</button></div>
        {rarePkg.image_preview && <img src={rarePkg.image_preview} alt="病灶" className="message-evidence-thumb" />}
        <EvidenceChart predictions={rarePkg.top_predictions} />
        <div className="message-evidence-stats"><EvidenceStats predictions={rarePkg.top_predictions} rag={rarePkg.rag} /></div>
      </div>}
      {message.run && <ExecutionCard run={{ ...message.run, running: false }} />}
      {message.pendingAction && <ApprovalCard action={message.pendingAction} busy={approving} onConfirm={() => onApprove(message, true)} onReject={() => onApprove(message, false)} />}
      {message.artifacts?.map((item) => <ArtifactCard key={item.document_id} item={item} token={token} onOpen={onOpen} />)}
      {message.caseId && <button className="inline-workspace-link" onClick={() => onOpen({ type: "case", id: message.caseId })}><Stethoscope size={15} />打开病例工作台<span>{message.caseId}</span></button>}
      {message.reportId && <button className="inline-workspace-link" onClick={() => onOpen({ type: "report", id: message.reportId })}><FilePenLine size={15} />打开可编辑报告<span>{message.reportId}</span></button>}
      <div className="message-meta"><span>{formatTime(message.createdAt)}</span>{message.role === "assistant" && <button onClick={copy}>{copied ? <Check size={12} /> : <Copy size={12} />}{copied ? "已复制" : "复制"}</button>}</div>
    </div>
  </article>;
}

function EmptyChat({ onPrompt, user }) {
  const prompts = [
    [Stethoscope, "整理病例", "从就诊信息和症状开始", "我需要准备哪些病例信息？"],
    [BookOpen, "查找资料", "比较疾病表现与检查依据", "检索色素痣与黑色素瘤的鉴别要点"],
    [FilePenLine, "开始分析", "添加图片，衔接病例记录", "创建新病例并执行联合分析"],
  ];
  return <section className="empty-chat"><div className="welcome-mark"><BrandMark size={48} /><span>信肤 · 工作台</span></div><h1>今天，从这里开始。</h1><p>随时提问，也可以从一份病例开始。</p><div className="prompt-grid">{prompts.map(([Icon, title, description, prompt]) => <button key={title} onClick={() => onPrompt(prompt)}><span className="prompt-icon"><Icon size={19} /></span><strong>{title}</strong><span className="prompt-description">{description}</span><ArrowUpRight className="prompt-arrow" size={15} /></button>)}</div><div className="welcome-note"><span /><span>记录、资料与下一步工作，都在这里。</span></div></section>;
}

function fileToBase64(file) { return new Promise((resolve) => { if (!file) { resolve(""); return; } const reader = new FileReader(); reader.onload = () => resolve(String(reader.result || "")); reader.onerror = () => resolve(""); reader.readAsDataURL(file); }); }

function EvidenceChart({ predictions }) {
  if (!predictions?.length) return null;
  const max = Math.max(...predictions.map((item) => Number(item.score || 0)));
  return <div className="evidence-chart"><div className="evidence-chart-head"><BarChart3 size={14} /><strong>候选方向</strong></div><div className="evidence-chart-bars">{predictions.map((item, index) => <div key={index} className="evidence-bar-row"><span>{LABELS[item.label?.toLowerCase()] || item.label}</span><div className="evidence-bar-track"><i style={{ width: `${Math.max(4, Number(item.score || 0) / max * 100)}%` }} /></div><b>{(Number(item.score || 0) * 100).toFixed(1)}%</b></div>)}</div></div>;
}

function EvidenceStats({ predictions, rag }) {
  const candidates = predictions || [];
  const items = rag?.items?.length || 0;
  return <div className="evidence-stats-grid"><div className="evidence-stat-card"><span>首项得分</span><strong>{(Number(candidates[0]?.score || 0) * 100).toFixed(1)}%</strong><small>{LABELS[candidates[0]?.label?.toLowerCase()] || candidates[0]?.label || "—"}</small></div><div className="evidence-stat-card"><span>候选方向</span><strong>{candidates.length} 项</strong><small>按得分排列</small></div><div className="evidence-stat-card"><span>相关资料</span><strong>{items} 条</strong><small>可展开查看</small></div><div className="evidence-stat-card"><span>资料状态</span><strong>{rag?.evidence_sufficient ? "已核对" : "待补充"}</strong><small>{rag?.evidence_sufficient ? "查看说明与依据" : "可继续查找资料"}</small></div></div>;
}

function EvidenceDetail({ evidence, onClose, onUse, onDeep, onDelete }) {
  if (!evidence) return null;
  return <div className="evidence-detail-overlay" onClick={onClose}><div className="evidence-detail-drawer" onClick={(e) => e.stopPropagation()}><header><div><span className="rare-kicker"><BookOpen size={13} />分析记录</span><strong>病灶与相关资料</strong><small>{MODE_LABELS[evidence.mode] || "分析记录"} · {formatTime(evidence.created_at || now())}</small></div><div className="evidence-detail-actions"><button onClick={() => onUse(evidence)}><Check size={14} />加入对话</button><button onClick={() => onDeep(evidence)}><ArrowUpRight size={14} />深度推理</button><button className="danger" onClick={() => { if (window.confirm(`确定删除证据包 ${evidence.package_id}？`)) onDelete(evidence); }}><Trash2 size={14} />删除</button><button className="close" onClick={onClose}><X size={16} /></button></div></header>
    <div className="evidence-detail-body">
      <div className="evidence-detail-left">
        {evidence.image_preview ? <div className="evidence-image-section"><div className="evidence-section-head"><ScanSearch size={14} /><strong>病灶图像</strong></div><img src={evidence.image_preview} alt="病灶" className="evidence-lesion-image" /></div> : <div className="evidence-image-section empty"><ScanSearch size={20} /><span>未添加图片</span></div>}
        {evidence.text && <div className="evidence-text-section"><div className="evidence-section-head"><FileText size={14} /><strong>输入文本</strong></div><p>{evidence.text}</p></div>}
      </div>
      <div className="evidence-detail-right">
        <EvidenceChart predictions={evidence.top_predictions} />
        <div className="evidence-section-head"><Activity size={14} /><strong>记录概览</strong></div>
        <EvidenceStats predictions={evidence.top_predictions} rag={evidence.rag} />
        <div className="evidence-section-head"><BookOpen size={14} /><strong>资料说明</strong></div>
        <div className="evidence-rag-answer"><Markdown>{evidence.rag?.answer || "暂无说明"}</Markdown></div>
        {(evidence.rag?.items || []).length > 0 && <div className="evidence-rag-items"><div className="evidence-section-head"><Layers3 size={14} /><strong>引用来源（{(evidence.rag?.items || []).length} 条）</strong></div>{(evidence.rag?.items || []).map((item, index) => <div key={index} className="evidence-rag-item"><span className="rag-item-title">{item.title || item.document_id || `证据 ${index + 1}`}</span><small>{item.source || item.document_source || "本地知识库"} · {item.evidence_status || ""}</small><p>{(item.content || item.snippet || "").slice(0, 200)}</p></div>)}</div>}
      </div>
    </div>
  </div></div>;
}

function RareDiseasePanel({ token, evidence, onRun, onUse, onDeep }) {
  const [mode, setMode] = useState("image_text"); const [text, setText] = useState(""); const [file, setFile] = useState(null); const [running, setRunning] = useState(false); const inputRef = useRef(null);
  async function run() { setRunning(true); try { const form = new FormData(); form.append("mode", mode); form.append("text", text); form.append("top_k", "5"); if (file) form.append("image", file); const result = await analyzeLesion(form, token); if (file) result.image_preview = await fileToBase64(file); onRun(result); } catch (err) { onRun({ error: friendlyError(err) }); } finally { setRunning(false); } }
  return <div className="rare-panel rare-panel-wide"><div className="rare-mode-tabs">{Object.entries(MODE_LABELS).map(([key, label]) => <button key={key} className={mode === key ? "active" : ""} onClick={() => setMode(key)}>{label}</button>)}</div><div className="rare-input-grid rare-input-grid-wide"><label className="rare-upload"><input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(event) => setFile(event.target.files?.[0] || null)} /><ImagePlus size={18} /><b>{file ? file.name : "选择病灶图片"}</b><small>JPG、PNG 或 WebP</small></label><textarea value={text} onChange={(event) => setText(event.target.value)} placeholder="输入病例、病程、罕见表现或希望核对的问题…" /></div><div className="rare-panel-actions"><button className="rare-run" onClick={run} disabled={running || (mode !== "text_only" && !file) || (mode !== "image_only" && mode !== "zero_shot" && !text.trim())}>{running ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />}{running ? "正在分析…" : "开始分析"}</button>{evidence && !evidence.error && <><button className="rare-use" onClick={() => onUse(evidence)}><Check size={14} />加入当前对话</button><button className="rare-deep" onClick={() => onDeep(evidence)}>深度推理 <ArrowUpRight size={13} /></button></>}</div>{evidence?.error && <p className="rare-error">{evidence.error}</p>}{evidence && !evidence.error && <div className="rare-result-strip"><span><b>分析记录</b> 已保存</span><span>{evidence.top_predictions?.length || 0} 个候选方向 · {evidence.rag?.items?.length || 0} 条资料</span><small>展开查看图片、说明与资料</small></div>}</div>;
}

function Composer({ inactive, value, onChange, onSend, onDocument, attachment, setAttachment, attachmentKind, setAttachmentKind, intakeState, busy, models, model, setModel, token, rareEvidence, onRareRun, onRareUse, onRareDeep, reasoningMode, setReasoningMode, lockedEvidenceId, evidencePackages }) {
  const fileRef = useRef(null);
  const documentRef = useRef(null);
  const composerRef = useRef(null);
  useEffect(() => { if (composerRef.current) { composerRef.current.style.height = "0px"; composerRef.current.style.height = `${Math.min(160, Math.max(42, composerRef.current.scrollHeight))}px`; } }, [value]);
  const [menuOpen, setMenuOpen] = useState(false); const [rareOpen, setRareOpen] = useState(false); const [pkgOpen, setPkgOpen] = useState(false);
  const previewUrl = useMemo(() => attachment ? URL.createObjectURL(attachment) : "", [attachment]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  function keyDown(event) { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent?.isComposing && !event.isComposing) { event.preventDefault(); onSend(); } }
  const modes = [{ id: "standard", label: "标准", icon: <MessageSquare size={13} />, tip: "日常问答与资料查询" }, { id: "enhanced", label: "增强", icon: <ScanSearch size={13} />, tip: "结合图片与病例分析" }, { id: "deep", label: "深度", icon: <BrainCircuit size={13} />, tip: "结合上下文，进一步查找依据" }];
  return <div className="composer-zone" inert={inactive}>{rareOpen && <RareDiseasePanel token={token} evidence={rareEvidence} onRun={onRareRun} onUse={(item) => { onRareUse(item); setRareOpen(false); }} onDeep={onRareDeep} />}
    {pkgOpen && (evidencePackages || []).length > 0 && <div className="pkg-import-panel"><div className="pkg-import-head"><BrainCircuit size={14} /><strong>选用分析记录</strong><button onClick={() => setPkgOpen(false)}><X size={14} /></button></div><div className="pkg-import-list">{evidencePackages.map((pkg) => <button key={pkg.package_id} className={`pkg-import-item ${pkg.package_id === lockedEvidenceId ? "active" : ""}`} onClick={() => { onRareUse(pkg); setReasoningMode("deep"); setPkgOpen(false); }}><div className="pkg-import-id">{formatTime(pkg.created_at || now())} · {MODE_LABELS[pkg.mode] || "图片分析"}</div><div className="pkg-import-top">{(pkg.top_predictions || []).slice(0, 2).map((item, i) => <span key={i}><b>{LABELS[item.label] || item.label}</b> {(Number(item.score) * 100).toFixed(0)}%</span>)}</div><small>{pkg.rag?.items?.length || 0} 条相关资料</small></button>)}</div></div>}
    <div className="composer">
      <div className="composer-mode-row">
        <div className="reasoning-mode-picker" role="radiogroup" aria-label="推理模式">
          {modes.map((item) => <button key={item.id} type="button" role="radio" aria-checked={reasoningMode === item.id} className={`mode-chip ${reasoningMode === item.id ? "active" : ""}`} title={item.tip} onClick={() => setReasoningMode(item.id)}>{item.icon}<span>{item.label}</span>{item.id === "deep" && lockedEvidenceId && <i className="lock-tag">已选记录</i>}</button>)}
        </div>
      </div>
      <textarea ref={composerRef} rows={1} aria-label="输入问题" placeholder={reasoningMode === "deep" ? (lockedEvidenceId ? "围绕选中的记录继续提问…" : "写下需要进一步查找依据的问题…") : reasoningMode === "enhanced" ? "添加病灶图片，写下症状与变化…" : "写下问题，或添加一份病例…"} value={value} onChange={(e) => onChange(e.target.value)} onKeyDown={keyDown} />{attachment && <div className="attachment-preview"><img src={previewUrl} alt="待分析病灶" /><div className="attachment-meta"><strong>{attachment.name}</strong><span>{attachmentKind === "lesion" ? "病灶图片" : "常规图片"} · {Math.round(attachment.size / 1024)} KB</span></div><button onClick={() => setAttachment(null)} title="移除附件"><X size={15} /></button></div>}<div className="composer-tools"><div className="plus-picker"><button className="tool-button plus-button" title="添加附件" onClick={() => setMenuOpen((v) => !v)}><Plus size={18} /></button>{menuOpen && <div className="plus-menu"><button onClick={() => { setAttachmentKind("normal"); fileRef.current?.click(); setMenuOpen(false); }}><ImagePlus size={15} /><span>常规图片<small>仅作为对话附件</small></span></button><button onClick={() => { setAttachmentKind("lesion"); fileRef.current?.click(); setMenuOpen(false); }}><ScanSearch size={15} /><span>皮肤病灶图片<small>自动进入增强模式</small></span></button><button onClick={() => { setRareOpen((v) => !v); setMenuOpen(false); }}><BrainCircuit size={15} /><span>病灶分析<small>查看图片与相关资料</small></span></button>{(evidencePackages || []).length > 0 && <button onClick={() => { setPkgOpen((v) => !v); setMenuOpen(false); }}><Archive size={15} /><span>选用分析记录<small>选用并切到深度模式</small></span></button>}</div>}</div><input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => { const file = e.target.files?.[0] || null; setAttachment(file); if (file && attachmentKind === "lesion") setReasoningMode("enhanced"); e.target.value = ""; }} /><input ref={documentRef} type="file" accept=".md,.txt,.json,.csv" hidden onChange={(e) => { const file = e.target.files?.[0]; if (file) onDocument(file); e.target.value = ""; }} /><button className="tool-button" title="上传可编辑文档" onClick={() => documentRef.current?.click()}><FileUp size={18} /></button><select aria-label="选择助手" value={model} onChange={(e) => setModel(e.target.value)}>{models.map((item) => <option key={item.id} value={item.id} disabled={!item.available}>{item.label}{item.available ? "" : " · 不可用"}</option>)}</select><button aria-label={busy ? "停止等待" : "发送消息"} className={`send-button ${busy ? "stop" : ""}`} onClick={onSend} disabled={!busy && !value.trim() && !attachment}>{busy ? <Square size={14} /> : <ArrowUp size={18} />}</button></div></div>
    <p className="composer-hint"><span>Shift + Enter 换行</span><span>Enter 发送</span></p>
  </div>;
}

function CaseView({ id, token, onDeleted }) {
  const [data, setData] = useState(null); const [error, setError] = useState("");
  const [editing, setEditing] = useState(false); const [form, setForm] = useState(null); const [saving, setSaving] = useState(false);
  useEffect(() => { let url = ""; Promise.all([
    request(`/v1/cases/${id}/patient`, {}, token), request(`/v1/cases/${id}/visual`, {}, token),
    request(`/v1/cases/${id}/triage`, {}, token), request(`/v1/cases/${id}/care-pathway`, {}, token),
    fetchBlob(`/v1/cases/${id}/image`, token),
  ]).then(([patient, visual, triage, care, blob]) => { url = URL.createObjectURL(blob); setData({ patient, visual, triage, care, url }); }).catch((err) => setError(friendlyError(err))).finally(() => {}); return () => { if (url) URL.revokeObjectURL(url); }; }, [id, token]);
  if (error && !data) return <div className="drawer-state error">{error}</div>;
  if (!data) return <div className="drawer-state"><LoaderCircle className="spin" />正在读取病例…</div>;
  async function saveCase() {
    setSaving(true); setError("");
    try { const patient = await request(`/v1/cases/${id}`, { method: "PUT", body: JSON.stringify(form) }, token); setData((old) => ({ ...old, patient })); setEditing(false); }
    catch (err) { setError(friendlyError(err)); } finally { setSaving(false); }
  }
  async function removeCase() {
    const confirmation = window.prompt(`删除病例会同时删除其图片、分析和报告。\n请输入 ${id} 确认：`);
    if (confirmation !== id) return;
    try { await request(`/v1/cases/${id}`, { method: "DELETE", body: JSON.stringify({ confirm_case_id: id }) }, token); onDeleted?.(); }
    catch (err) { setError(friendlyError(err)); }
  }
  function startEdit() { setForm({ name: data.patient.name, sex: data.patient.sex, age: data.patient.age, outpatient_no: data.patient.outpatient_no, phone: data.patient.phone || "", medical_history: data.patient.medical_history || "", allergy_history: data.patient.allergy_history || "", lesion_site: data.patient.lesion_site, chief_complaint: data.patient.chief_complaint, clinical_profile: data.patient.clinical_profile }); setEditing(true); }
  if (editing) return <div className="case-edit"><div className="editor-meta"><div><p className="eyebrow">编辑病例</p><h2>{id}</h2><span>修改患者和结构化病例信息</span></div><div><button onClick={() => setEditing(false)}>取消</button><button className="primary" disabled={saving} onClick={saveCase}>{saving ? <LoaderCircle className="spin" size={14} /> : <Check size={14} />}保存</button></div></div><div className="case-form">
    <label>患者姓名<input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label><label>门诊号<input value={form.outpatient_no} onChange={(e) => setForm({ ...form, outpatient_no: e.target.value })} /></label><label>性别<select value={form.sex} onChange={(e) => setForm({ ...form, sex: e.target.value, clinical_profile: { ...form.clinical_profile, gender: e.target.value === "男" ? "MALE" : "FEMALE" } })}><option>男</option><option>女</option></select></label><label>年龄<input type="number" value={form.age} onChange={(e) => setForm({ ...form, age: Number(e.target.value), clinical_profile: { ...form.clinical_profile, age: Number(e.target.value) } })} /></label><label>电话<input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} /></label><label>病灶部位<input value={form.lesion_site} onChange={(e) => setForm({ ...form, lesion_site: e.target.value })} /></label><label className="wide">主诉<textarea value={form.chief_complaint} onChange={(e) => setForm({ ...form, chief_complaint: e.target.value })} /></label><fieldset className="wide"><legend>结构化症状</legend>{[["itch","瘙痒"],["hurt","疼痛"],["bleed","出血"],["grew","近期增大"],["changed","近期变化"],["skin_cancer_history","皮肤癌病史"],["cancer_history","其他癌症病史"]].map(([key,label]) => <label className="check" key={key}><input type="checkbox" checked={Boolean(form.clinical_profile?.[key])} onChange={(e) => setForm({ ...form, clinical_profile: { ...form.clinical_profile, [key]: e.target.checked } })} />{label}</label>)}</fieldset><label>长径（mm）<input type="number" step="0.1" value={form.clinical_profile?.diameter_1 || ""} onChange={(e) => setForm({ ...form, clinical_profile: { ...form.clinical_profile, diameter_1: Number(e.target.value) } })} /></label><label>短径（mm）<input type="number" step="0.1" value={form.clinical_profile?.diameter_2 || ""} onChange={(e) => setForm({ ...form, clinical_profile: { ...form.clinical_profile, diameter_2: Number(e.target.value) } })} /></label>
  </div>{error && <div className="form-error">{error}</div>}</div>;
  return <div className="case-view"><div className="case-visual"><img src={data.url} alt="病灶" /><span>病灶图片</span><div className="case-actions"><button onClick={startEdit}><Pencil size={14} />编辑病例</button><button className="danger" onClick={removeCase}><Trash2 size={14} />删除病例</button></div>{error && <div className="form-error">{error}</div>}</div><div className="case-details">
    <div className="detail-card identity"><p className="eyebrow">患者与就诊信息</p><h2>{data.patient.name}</h2><div className="identity-grid"><span>门诊号<strong>{data.patient.outpatient_no}</strong></span><span>性别 / 年龄<strong>{data.patient.sex} · {data.patient.age} 岁</strong></span><span>病灶部位<strong>{data.patient.lesion_site}</strong></span><span>病例编号<strong>{id}</strong></span></div><p>{data.patient.chief_complaint}</p></div>
    <div className="detail-card"><p className="eyebrow">联合分析候选</p><div className="prediction-list">{data.visual.top_predictions?.map((item, index) => <div key={item.label}><span>{index + 1}</span><strong>{LABELS[item.label?.toLowerCase()] || item.label}</strong><i><b style={{ width: `${Math.max(2, item.score * 100)}%` }}></b></i><em>{(item.score * 100).toFixed(1)}%</em></div>)}</div></div>
    <div className="detail-card risk-card"><p className="eyebrow">风险与照护路径</p><h3>{LEVELS[data.triage.level] || data.triage.level}</h3><p>{data.care.action}</p><ul>{data.care.recommended_next_steps?.map((item) => <li key={item}>{item}</li>)}</ul></div>
  </div></div>;
}

function Library({ type, token, onOpen }) {
  const [items, setItems] = useState(null); const [query, setQuery] = useState(""); const [folders, setFolders] = useState([]); const [folder, setFolder] = useState(""); const [version, setVersion] = useState(0); const uploadRef = useRef(null);
  const [notice, setNotice] = useState("");
  const [filters, setFilters] = useState({ sex: "", triage_level: "", diagnosis: "", min_age: "", max_age: "", lesion_site: "" });
  const resourceType = type === "cases" ? "case" : type === "reports" ? "report" : "document";
  useEffect(() => { request("/v1/workspace/folders", {}, token).then(setFolders).catch(() => setFolders([])); }, [token, version]);
  useEffect(() => {
    const params = new URLSearchParams(); if (folder) params.set("folder_id", folder);
    if (type === "cases") Object.entries(filters).forEach(([key,value]) => { if (value !== "") params.set(key,value); });
    const endpoint = type === "cases" ? "/v1/cases" : type === "reports" ? "/v1/reports" : "/v1/agent/documents";
    request(`${endpoint}?${params}`, {}, token).then(setItems).catch((err) => { setItems([]); setNotice(friendlyError(err)); });
  }, [type, token, folder, filters, version]);
  const filtered = (items || []).filter((item) => JSON.stringify(item).toLowerCase().includes(query.toLowerCase()));
  const folderNames = Object.fromEntries(folders.map((item) => [item.folder_id, item.name]));
  async function createFolder() { const name = window.prompt("文件夹名称"); if (!name?.trim()) return; await request("/v1/workspace/folders", { method: "POST", body: JSON.stringify({ name: name.trim(), parent_id: null }) }, token); setVersion((n) => n + 1); }
  async function deleteFolder(item) { if (!window.confirm(`删除文件夹“${item.name}”？其中的病例、报告和文档会保留并回到“全部内容”。`)) return; await request(`/v1/workspace/folders/${item.folder_id}`, { method: "DELETE" }, token); if (folder === item.folder_id) setFolder(""); setVersion((n) => n + 1); }
  async function move(item, folderId) { if (folderId === "__choose") return; const id = workspaceResourceId(type, item); try { await request("/v1/workspace/items", { method: "PUT", body: JSON.stringify({ resource_type: resourceType, resource_id: id, folder_id: folderId === "__root" ? null : folderId }) }, token); setNotice(folderId === "__root" ? "已移回根目录" : `已移入文件夹“${folderNames[folderId]}”`); setVersion((n) => n + 1); } catch (err) { setNotice(`移动失败：${friendlyError(err)}`); } }
  async function remove(item, event) { event.stopPropagation(); const id = workspaceResourceId(type, item); if (type === "cases") { const confirmation = window.prompt(`删除病例会同时删除其图片、分析和报告。\n请输入 ${id} 确认：`); if (confirmation !== id) return; } else if (!window.confirm(`确定删除“${item.title || item.filename || id}”？此操作不可撤销。`)) return; const endpoint = type === "cases" ? `/v1/cases/${id}` : type === "reports" ? `/v1/reports/${id}` : `/v1/agent/documents/${id}`; const options = type === "cases" ? { method: "DELETE", body: JSON.stringify({ confirm_case_id: id }) } : { method: "DELETE" }; try { await request(endpoint, options, token); setNotice("删除完成"); setVersion((n) => n + 1); } catch (err) { setNotice(`删除失败：${friendlyError(err)}`); } }
  async function upload(event) { const file = event.target.files?.[0]; if (!file) return; const form = new FormData(); form.append("file", file); form.append("conversation_id", "workspace"); form.append("folder_id", folder); await request("/v1/agent/documents/upload", { method: "POST", body: form }, token); event.target.value = ""; setVersion((n) => n + 1); }
  const descriptions = { cases: "查看就诊记录、图片与分析结果。", reports: "整理报告，查看历史版本。", documents: "收集工作资料，随时编辑与归档。" };
  return <div className="library-layout"><aside className="folder-panel"><span className="folder-heading">目录</span><button className={!folder ? "active" : ""} onClick={() => setFolder("")}><Archive size={15} />全部内容</button>{folders.map((item) => <div className={`folder-entry ${folder === item.folder_id ? "active" : ""}`} key={item.folder_id}><button onClick={() => setFolder(item.folder_id)}><Archive size={13} />{item.name}</button><button title="删除文件夹" onClick={() => deleteFolder(item)}><X size={12} /></button></div>)}<button className="new-folder" onClick={createFolder}><Plus size={14} />新建文件夹</button></aside><div className="library"><div className="library-overview"><div><h1>{type === "cases" ? "病例库" : type === "reports" ? "报告库" : "文档库"}</h1><p className="library-description">{descriptions[type]}</p></div><div className="library-total"><strong>{items === null ? "—" : filtered.length}</strong><span>份记录</span></div></div>{notice && <div className="library-notice"><Check size={14} />{notice}</div>}<div className="library-toolbar"><div className="library-search"><Search size={16} /><input placeholder={`搜索${type === "cases" ? "病例" : type === "reports" ? "报告" : "文档"}`} value={query} onChange={(e) => setQuery(e.target.value)} /></div>{type === "documents" && <><input hidden ref={uploadRef} type="file" accept=".md,.txt,.json,.csv" onChange={upload} /><button className="upload-document" onClick={() => uploadRef.current?.click()}><Plus size={15} />上传文档</button></>}</div>{type === "cases" && <div className="structured-filters"><select value={filters.sex} onChange={(e) => setFilters({ ...filters, sex: e.target.value })}><option value="">全部性别</option><option>男</option><option>女</option></select><input placeholder="最小年龄" type="number" value={filters.min_age} onChange={(e) => setFilters({ ...filters, min_age: e.target.value })} /><input placeholder="最大年龄" type="number" value={filters.max_age} onChange={(e) => setFilters({ ...filters, max_age: e.target.value })} /><input placeholder="病灶部位" value={filters.lesion_site} onChange={(e) => setFilters({ ...filters, lesion_site: e.target.value })} /><input placeholder="Top-1 候选" value={filters.diagnosis} onChange={(e) => setFilters({ ...filters, diagnosis: e.target.value })} /><select value={filters.triage_level} onChange={(e) => setFilters({ ...filters, triage_level: e.target.value })}><option value="">全部风险</option><option value="LOCAL_REVIEW">基层评估</option><option value="REFERRAL">建议转诊</option><option value="PRIORITY_REFERRAL">优先转诊</option></select></div>}{items === null ? <div className="drawer-state"><LoaderCircle className="spin" />正在读取…</div> : <div className="library-list">{!filtered.length && <div className="library-empty"><Archive /><strong>{query ? "没有找到相关内容" : "这里还没有记录"}</strong><span>{query ? "试试其他关键词。" : type === "documents" ? "上传一份资料，开始整理。" : "新记录会出现在这里。"}</span></div>}{filtered.map((item) => { const id = workspaceResourceId(type, item); const location = item.folder_id ? folderNames[item.folder_id] : "未归档"; return <div className="library-row" key={id} onClick={() => onOpen({ type: type === "cases" ? "case" : type === "reports" ? "report" : "document", id, data: item })}><span className="library-icon">{type === "cases" ? <Stethoscope /> : type === "reports" ? <FileText /> : <FilePenLine />}</span><span><strong>{type === "cases" ? (item.patient_name || item.case_id) : (item.title || item.filename || item.patient_name || id)}</strong><small>{type === "cases" ? `${item.outpatient_no || "未关联门诊号"} · ${item.lesion_site || "未填写部位"} · ${LABELS[item.top_diagnosis] || item.top_diagnosis || "未分析"}` : type === "reports" ? `${item.patient_name || ""} · ${item.outpatient_no || ""} · ${REPORT_STATUS[item.status] || item.status}` : `${item.filename} · 第 ${item.revision} 版`}<b className="folder-badge">{location}</b></small></span><select value={item.folder_id || "__root"} onClick={(e) => e.stopPropagation()} onChange={(e) => move(item,e.target.value)}><option value="__root">未归档</option>{folders.map((f) => <option value={f.folder_id} key={f.folder_id}>{f.name}</option>)}</select><em>{formatTime(item.updated_at || item.created_at)}</em><button className="row-delete" title="删除" onClick={(event) => remove(item, event)}><Trash2 size={14} /></button></div>; })}</div>}</div></div>;
}

function ReportEditor({ id, token, onDeleted }) {
  const [report, setReport] = useState(null); const [content, setContent] = useState(""); const [state, setState] = useState("");
  const [viewMode, setViewMode] = useState("preview");
  useEffect(() => { request(`/v1/reports/${id}`, {}, token).then((item) => { setReport(item); setContent(item.content); }).catch((err) => setState(friendlyError(err))); }, [id, token]);
  async function save() { setState("正在保存…"); try { const item = await request(`/v1/reports/${id}`, { method: "PUT", body: JSON.stringify({ content, status: report.status || "draft" }) }, token); setReport(item); setState("已保存新版本"); } catch (err) { setState(friendlyError(err)); } }
  async function download() { const blob = await fetchBlob(`/v1/reports/${id}/download`, token); const url = URL.createObjectURL(blob), a = document.createElement("a"); a.href = url; a.download = report?.filename || `${id}.md`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 500); }
  async function remove() { if (!window.confirm(`确定删除报告“${report.filename}”？病例和图片将保留。`)) return; try { await request(`/v1/reports/${id}`, { method: "DELETE" }, token); onDeleted?.(); } catch (err) { setState(friendlyError(err)); } }
  if (!report) return <div className="drawer-state">{state || <><LoaderCircle className="spin" />正在读取报告…</>}</div>;
  return <div className="editor-view"><div className="editor-meta"><div><p className="eyebrow">在线报告</p><h2>{report.filename}</h2><span>{report.patient_name} · {report.outpatient_no} · {REPORT_STATUS[report.status] || report.status}</span></div><div><span className="editor-tabs"><button className={viewMode === "edit" ? "active" : ""} onClick={() => setViewMode("edit")}><Pencil size={14} />编辑</button><button className={viewMode === "preview" ? "active" : ""} onClick={() => setViewMode("preview")}><Eye size={14} />预览</button></span><button onClick={download}><Download size={15} />下载</button><button className="danger-button" onClick={remove}><Trash2 size={14} />删除</button><button className="primary" onClick={save}><Check size={15} />保存修改</button></div></div>{viewMode === "edit" ? <textarea value={content} onChange={(e) => setContent(e.target.value)} /> : <div className="markdown-preview"><Markdown token={token}>{content}</Markdown></div>}<div className="save-state">{state}</div></div>;
}

function evidenceImageMarkdown(content, title, rarePackages) {
  const text = String(content || "");
  if (/!\[/.test(text)) return text;
  const pkgId = (String(title || "") + "\n" + text.slice(0, 800)).match(/RARE-\d{8}-\d{6}-[A-F0-9]{6}/i)?.[0];
  if (!pkgId) return text;
  const pkg = (rarePackages || []).find((item) => item.package_id === pkgId);
  if (!pkg?.image_preview) return text;
  return `## 患者病灶图像\n\n![患者病灶图片](evidence-image://${pkgId})\n\n${text}`;
}

function DocumentEditor({ item, token, rarePackages, onUpdated, onDeleted }) {
  const [title, setTitle] = useState(item.title); const [content, setContent] = useState(item.content || ""); const [state, setState] = useState("");
  const [viewMode, setViewMode] = useState("preview");
  const [revision, setRevision] = useState(item.revision || 1);
  const [loading, setLoading] = useState(true); const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setLoading(true); setLoaded(false); setState("");
    request(`/v1/agent/documents/${encodeURIComponent(item.document_id)}`, {}, token)
      .then((doc) => { if (cancelled) return; setContent(doc.content || ""); setTitle(doc.title || item.title); setRevision(doc.revision || 1); setLoaded(true); })
      .catch((err) => { if (!cancelled) setState(friendlyError(err)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [item.document_id, token]);
  const preview = useMemo(() => evidenceImageMarkdown(content, item.title, rarePackages), [content, item.title, rarePackages]);
  async function save() { setState("正在保存…"); try { const saved = await request(`/v1/agent/documents/${item.document_id}`, { method: "PUT", body: JSON.stringify({ title, content }) }, token); setRevision(saved.revision); setState(`已保存第 ${saved.revision} 版`); onUpdated?.(saved); } catch (err) { setState(friendlyError(err)); } }
  async function remove() { if (!window.confirm(`确定删除文档“${item.filename}”？关联病例和报告将保留。`)) return; try { await request(`/v1/agent/documents/${item.document_id}`, { method: "DELETE" }, token); onDeleted?.(); } catch (err) { setState(friendlyError(err)); } }
  if (loading) return <div className="drawer-state"><LoaderCircle className="spin" />正在读取文档…</div>;
  if (!loaded) return <div className="drawer-state">{state}</div>;
  return <div className="editor-view"><div className="editor-meta"><div><p className="eyebrow">工作文档</p><input className="editor-title" value={title} onChange={(e) => setTitle(e.target.value)} /><span>{item.filename} · 第 {revision} 版</span></div><div><span className="editor-tabs"><button className={viewMode === "edit" ? "active" : ""} onClick={() => setViewMode("edit")}><Pencil size={14} />编辑</button><button className={viewMode === "preview" ? "active" : ""} onClick={() => setViewMode("preview")}><Eye size={14} />预览</button></span><button className="danger-button" onClick={remove}><Trash2 size={14} />删除</button><button className="primary" onClick={save}><Check size={15} />保存新版本</button></div></div>{viewMode === "edit" ? <textarea value={content} onChange={(e) => setContent(e.target.value)} /> : <div className="markdown-preview"><Markdown token={token} rarePackages={rarePackages}>{preview}</Markdown></div>}<div className="save-state">{state}</div></div>;
}

function RareDiseaseCenter({ token, onUse, onDeep, onPersist, evidencePackages, selectedRareEvidenceId, onDeleteEvidence }) {
  const [evidence, setEvidence] = useState(null); const [notice, setNotice] = useState(""); const [detailPkg, setDetailPkg] = useState(null);
  function handleRun(result) { setEvidence(result); onPersist?.(result); if (result?.error) setNotice(result.error); else { setNotice(`分析记录已保存`); setDetailPkg(result); } }
  return <div className="rare-center rare-center-wide">
    <div className="rare-center-hero"><div><span className="eyebrow">图片与病例</span><h2>病灶分析</h2><p>添加图片与描述，整理可能的方向和相关依据。</p></div></div>
    {notice && <div className="rare-center-notice"><Check size={14} />{notice}</div>}
    {evidence && !evidence.error && <div className="rare-center-grid"><div className="rare-center-card"><span>证据包</span><strong>{formatTime(evidence.created_at || now())}</strong><small>{MODE_LABELS[evidence.mode] || "分析记录"}</small></div><div className="rare-center-card"><span>候选方向</span><strong>{evidence.top_predictions?.length || 0} 个</strong><small>{(evidence.top_predictions || []).map((item) => `${item.label} ${(Number(item.score) * 100).toFixed(1)}%`).join(" · ")}</small></div><div className="rare-center-card"><span>相关资料</span><strong>{evidence.rag?.evidence_sufficient ? "覆盖充分" : "需要补充"}</strong><small>{"相关资料"} · {evidence.rag?.items?.length || 0} 条参考</small></div></div>}
    <div className="rare-center-columns">
      <RareDiseasePanel token={token} evidence={evidence} onRun={handleRun} onUse={(item) => { onUse?.(item); setNotice(`已加入当前对话`); }} onDeep={onDeep} />
      <div className="rare-history-section">
        <div className="rare-history-head"><BrainCircuit size={15} /><strong>分析记录</strong><small>点击卡片查看详情</small></div>
        {(evidencePackages || []).length ? <div className="rare-history-list">{evidencePackages.map((pkg) => <button key={pkg.package_id} className={`rare-history-item ${pkg.package_id === selectedRareEvidenceId ? "active" : ""}`} onClick={() => setDetailPkg(pkg)}><div className="rare-history-id">{formatTime(pkg.created_at || now())} · {MODE_LABELS[pkg.mode] || "病灶分析"}</div><div className="rare-history-top">{(pkg.top_predictions || []).slice(0, 3).map((item, i) => <span key={i}><b>{LABELS[item.label] || item.label}</b> {(Number(item.score) * 100).toFixed(0)}%</span>)}</div><div className="rare-history-meta"><span>{MODE_LABELS[pkg.mode] || "分析记录"}</span><span>{pkg.rag?.evidence_sufficient ? "证据充分" : "证据不足"}</span></div>{pkg.image_preview && <img src={pkg.image_preview} alt="病灶" className="rare-history-thumb" />}{pkg.package_id === selectedRareEvidenceId && <i className="rare-history-tag">当前选用</i>}</button>)}</div> : <div className="rare-history-empty"><Archive size={22} /><span>暂无分析记录</span><small>添加一张图片，开始第一份记录。</small></div>}
      </div>
    </div>
    {detailPkg && <EvidenceDetail evidence={detailPkg} onClose={() => setDetailPkg(null)} onUse={(item) => { onUse?.(item); setDetailPkg(null); }} onDeep={(item) => { onDeep?.(item); setDetailPkg(null); }} onDelete={(item) => { onDeleteEvidence?.(item); setDetailPkg(null); setNotice(`分析记录已删除`); }} />}
  </div>;
}

class WorkspaceErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error) { console.error("Workspace render error", error); }
  render() { if (!this.state.error) return this.props.children; return <div className="drawer-state error"><strong>页面暂时无法显示</strong><p>请重新打开此页面。</p><button onClick={() => this.setState({ error: null })}>重试</button></div>; }
}

function WorkspaceDrawer({ view, token, onClose, onOpen, onDocumentUpdated, onRareUse, onRarePersist, onRareDeep, evidencePackages, selectedRareEvidenceId, onDeleteEvidence }) {
  if (!view) return null;
  const titles = { cases: "病例库", reports: "报告库", documents: "文档库", knowledge: "资料与研究", rare: "病灶分析", case: "病例工作台", report: "报告编辑器", document: "工作文档" };
  return <div className="workspace-drawer"><header><button title="返回对话" onClick={onClose}><ChevronLeft size={18} /></button><strong>{titles[view.type]}</strong><span>{view.type === "case" ? "就诊记录" : view.type === "report" || view.type === "document" ? "编辑与预览" : ""}</span><button className="drawer-x" title="关闭页面" onClick={onClose}><X size={18} /></button></header><main className={view.type === "knowledge" ? "workspace-knowledge" : undefined} key={`${view.type}-${view.id || ""}`}>
    {view.type === "case" && <CaseView id={view.id} token={token} onDeleted={onClose} />}
    {(view.type === "cases" || view.type === "reports" || view.type === "documents") && <Library type={view.type} token={token} onOpen={onOpen} />}
    {view.type === "knowledge" && <React.Suspense fallback={<div className="drawer-state"><LoaderCircle className="spin" />正在打开资料…</div>}><KnowledgeCenter token={token} /></React.Suspense>}
    {view.type === "rare" && <WorkspaceErrorBoundary><RareDiseaseCenter token={token} onUse={onRareUse} onDeep={onRareDeep} onPersist={onRarePersist} evidencePackages={evidencePackages} selectedRareEvidenceId={selectedRareEvidenceId} onDeleteEvidence={onDeleteEvidence} /></WorkspaceErrorBoundary>}
    {view.type === "report" && <ReportEditor id={view.id} token={token} onDeleted={onClose} />}
    {view.type === "document" && <WorkspaceErrorBoundary><DocumentEditor item={view.data} token={token} rarePackages={evidencePackages} onUpdated={onDocumentUpdated} onDeleted={onClose} /></WorkspaceErrorBoundary>}
  </main></div>;
}

function answerFrom(payload) {
  const result = payload.result || {};
  let text = payload.message || "任务已完成。";
  if (result.kind === "chat" && result.chat?.response) text = result.chat.response;
  if (result.kind === "analysis" && result.analysis) {
    const analysis = result.analysis;
    const candidates = (analysis.visual?.top_predictions || []).slice(0, 3).map((item, index) => `${index + 1}. **${LABELS[item.label?.toLowerCase()] || item.label}** · ${(item.score * 100).toFixed(1)}%`).join("\n");
    const evidence = (analysis.guidelines || []).slice(0, 5).map((item) => `- ${item.title} · ${item.evidence_status || "待核验"} · 证据 ${item.id}`).join("\n");
    text += `\n\n### 联合分析结果\n${candidates || "暂无候选"}\n\n**风险分级**：${LEVELS[analysis.triage?.level] || analysis.triage?.level || "待复核"}\n\n**证据解释**\n${analysis.rag?.answer || "本次尚未完成证据解释。"}\n\n**本次检索资料**\n${evidence || "当前未检索到相关证据"}`;
  }
  return text;
}

export default function App() {
  const [token, setToken] = useState(() => browserStorage.getItem(TOKEN_KEY) || "");
  const [user, setUser] = useState(null); const [checking, setChecking] = useState(Boolean(token));
  const [threads, setThreads] = useState([]); const [activeId, setActiveId] = useState("");
  const [rarePackages, setRarePackages] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(() => typeof window === "undefined" || window.innerWidth > 860); const [view, setView] = useState(null);
  const [input, setInput] = useState(""); const [attachment, setAttachment] = useState(null); const [attachmentKind, setAttachmentKind] = useState("normal");
  const [models, setModels] = useState([]); const [model, setModel] = useState("auto");
  const [reasoningMode, setReasoningMode] = useState("standard");
  const [detailPkg, setDetailPkg] = useState(null);
  const [liveRun, setLiveRun] = useState(null); const [approving, setApproving] = useState(false);
  const abortRef = useRef(null); const bottomRef = useRef(null);
  const active = threads.find((item) => item.id === activeId) || threads[0];

  useEffect(() => {
    const expired = () => { abortRef.current?.abort(); browserStorage.removeItem(TOKEN_KEY); setToken(""); setUser(null); };
    window.addEventListener("xinfu:unauthorized", expired);
    return () => window.removeEventListener("xinfu:unauthorized", expired);
  }, []);

  useEffect(() => { if (!token) { setChecking(false); return; } authApi.me(token).then(setUser).catch(() => { browserStorage.removeItem(TOKEN_KEY); setToken(""); }).finally(() => setChecking(false)); }, [token]);
  useEffect(() => { if (!user) return; const loaded = loadThreads(user.id); const stored = loadRarePackages(user.id); const merged = [...stored]; loaded.forEach((t) => { (t.rareEvidencePackages || []).forEach((p) => { if (!merged.some((e) => e.package_id === p.package_id)) merged.push(p); }); delete t.rareEvidencePackages; }); setRarePackages(merged); setThreads(loaded); setActiveId(loaded[0].id); request("/v1/chat/models", {}, token).then((items) => { setModels(items); const first = items.find((item) => item.available); if (first) setModel(first.id); }).catch(() => setModels([{ id: "auto", label: "自动选择", available: true }])); }, [user, token]);
  useEffect(() => { if (user && threads.length) browserStorage.setItem(threadKey(user.id), JSON.stringify(threads)); }, [threads, user]);
  useEffect(() => { if (user) { try { browserStorage.setItem(rareKey(user.id), JSON.stringify(rarePackages)); } catch { /* 预览图过大超出配额时跳过持久化 */ } } }, [rarePackages, user]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.messages, liveRun]);

  async function handleLogin(payload) { browserStorage.setItem(TOKEN_KEY, payload.access_token); setToken(payload.access_token); setUser(payload.user); }
  async function logout() { try { await authApi.logout(token); } catch { /* local logout still succeeds */ } browserStorage.removeItem(TOKEN_KEY); setToken(""); setUser(null); setThreads([]); setRarePackages([]); }
  function updateActive(updater) { setThreads((items) => items.map((item) => item.id === active.id ? updater(item) : item)); }
  function addMessage(message) { updateActive((item) => ({ ...item, title: item.messages.length === 0 && message.role === "user" ? titleFrom(message.content) : item.title, updatedAt: now(), messages: [...item.messages, { id: uid(), createdAt: now(), ...message }] })); }
  function newChat() { if (window.innerWidth <= 860) setSidebarOpen(false); const item = freshThread(); setThreads((items) => [item, ...items]); setActiveId(item.id); setView(null); setAttachment(null); setInput(""); }
  function documentUpdated(saved) {
    setView((current) => current?.type === "document" && current.data?.document_id === saved.document_id ? { ...current, data: saved } : current);
    setThreads((items) => items.map((thread) => ({ ...thread, messages: thread.messages.map((message) => message.artifacts?.some((item) => item.document_id === saved.document_id) ? { ...message, artifacts: message.artifacts.map((item) => item.document_id === saved.document_id ? { ...item, ...saved } : item) } : message) })));
  }
  function renameThread(id) { const item = threads.find((thread) => thread.id === id); const title = window.prompt("输入新的对话名称", item?.title || ""); if (title?.trim()) setThreads((items) => items.map((thread) => thread.id === id ? { ...thread, title: title.trim(), updatedAt: now() } : thread)); }
  function deleteThread(id) { if (!window.confirm("删除该对话？病例、报告与证据包不会被删除。")) return; const next = threads.filter((item) => item.id !== id); if (!next.length) next.push(freshThread()); setThreads(next); if (activeId === id) setActiveId(next[0].id); }

  async function pollRun(runId, signal) {
    const deadline = Date.now() + 20 * 60 * 1000;
    while (Date.now() < deadline) {
      if (signal.aborted) throw new DOMException("已停止", "AbortError");
      const payload = await request(`/v1/agent/runs/${runId}`, { signal }, token);
      setLiveRun({ ...payload, running: payload.status === "RUNNING" });
      if (payload.status !== "RUNNING") return payload;
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
    throw new Error("任务执行超时");
  }

  async function uploadDocument(file) {
    if (!active || !file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("conversation_id", active.id);
    form.append("case_id", active.caseId || "");
    try {
      const document = await request("/v1/agent/documents/upload", { method: "POST", body: form }, token);
      const isClinicalIntake = looksLikeClinicalIntakeDocument(document.content);
      if (isClinicalIntake && !active.caseId) {
        updateActive((item) => {
          const block = `[病例资料文件：${file.name}]\n${document.content}`;
          const intakeDraft = [item.intakeDraft, block].filter(Boolean).join("\n\n").slice(-6000);
          return { ...item, intakeDraft, preparedIntake: null, intakeState: "COLLECTING", updatedAt: now() };
        });
      }
      addMessage({ role: "user", content: `[已上传文档] ${file.name}` });
      addMessage({
        role: "assistant",
        content: isClinicalIntake && !active.caseId
          ? "已读取病例信息，附病灶图片后即可开始分析。"
          : "文档已存入当前对话。",
        artifacts: [document],
      });
    } catch (err) {
      addMessage({ role: "assistant", operationError: true, content: `文档上传失败：${friendlyError(err)}` });
    }
  }

  async function runRareInference(payload) {
    if (payload.error) { addMessage({ role: "assistant", content: friendlyError({message: payload.error}, "暂时无法完成分析，请稍后重试。") }); return payload; }
    setRarePackages((items) => [...items.filter((entry) => entry.package_id !== payload.package_id), payload]);
    updateActive((item) => ({ ...item, selectedRareEvidenceId: payload.package_id, updatedAt: now() }));
    const topk = (payload.top_predictions || []).map((item, index) => `${index + 1}. **${item.label}** · ${(Number(item.score) * 100).toFixed(1)}%`).join("\n");
    addMessage({ role: "assistant", content: `### 病灶分析结果\n\n${topk || "暂无候选"}\n\n**资料说明**：${payload.rag?.answer || "暂暂无说明"}`, rareEvidenceId: payload.package_id });
    return payload;
  }

  function persistRarePackage(result) {
    if (!result || result.error) return;
    setRarePackages((items) => [...items.filter((entry) => entry.package_id !== result.package_id), result]);
  }

  function useRareEvidence(payload) {
    persistRarePackage(payload);
    updateActive((item) => ({ ...item, selectedRareEvidenceId: payload.package_id, updatedAt: now() }));
    addMessage({ role: "assistant", content: `已引用证据包 \`${payload.package_id}\`。`, rareEvidenceId: payload.package_id });
  }

  function deleteRareEvidence(payload) {
    setRarePackages((items) => items.filter((entry) => entry.package_id !== payload.package_id));
    updateActive((item) => item.selectedRareEvidenceId === payload.package_id ? { ...item, selectedRareEvidenceId: null, updatedAt: now() } : item);
  }

  async function send() {
    if (liveRun?.running) { abortRef.current?.abort(); return; }
    const text = input.trim(); if ((!text && !attachment) || !active) return;
    const file = attachment; const history = conversationHistory(active.messages);
    setInput(""); addMessage({ role: "user", content: text || "[已附加病灶图片和病例描述]" });
    const controller = new AbortController(); abortRef.current = controller; setLiveRun({ running: true, events: [] });
    try {
      // 增强模式 + 附件 lesion：自动走罕见病增强推理
      if (reasoningMode === "enhanced" && file && attachmentKind === "lesion") {
        const form = new FormData(); form.append("mode", text ? "image_text" : "image_only"); form.append("text", text); form.append("top_k", "5"); form.append("image", file);
        const rareResult = await analyzeLesion(form, token, controller.signal);
        rareResult.image_preview = await fileToBase64(file);
        await runRareInference(rareResult); setAttachment(null); setAttachmentKind("normal"); setLiveRun(null); return;
      }
      // 深度问答可独立使用；证据包作为可选上下文，附图仍走病例处理流程。
      if (reasoningMode === "deep" && !file) {
        const lockedEvidence = rarePackages.find((item) => item.package_id === active.selectedRareEvidenceId);
        const deepContext = lockedEvidence ? `\n\n[深度推理 · 锁定证据包 ${lockedEvidence.package_id}]\nTop-K：${(lockedEvidence.top_predictions || []).map((item) => `${item.label} ${(Number(item.score) * 100).toFixed(1)}%`).join("；")}\nRAG：${lockedEvidence.rag?.answer || "无"}\n证据充分性：${lockedEvidence.rag?.evidence_sufficient ? "充分" : "不足"}\n请围绕该证据包进行 agentic 检索和深度推理，引用本地证据，证据不足时明确指出。` : "";
        const deepMessage = ((text || "请进一步核对问题的依据与不确定性") + deepContext).slice(0, 4800);
        const form = new FormData(); form.append("message", deepMessage); form.append("case_id", active.caseId || ""); form.append("model", model); form.append("detail_level", "detailed"); form.append("reasoning_mode", "deep"); form.append("conversation_id", active.id); form.append("history_json", JSON.stringify(history));
        const initial = await request("/v1/agent/runs/dispatch", { method: "POST", body: form, signal: controller.signal }, token);
        const payload = await pollRun(initial.run_id, controller.signal);
        const artifacts = payload.result?.kind === "chat" ? payload.result.chat?.artifacts || [] : [];
        addMessage({ role: "assistant", content: answerFrom(payload), run: payload, pendingAction: payload.pending_action, runId: payload.run_id, caseId: payload.case_id, artifacts });
        if (payload.case_id) updateActive((item) => ({ ...item, caseId: payload.case_id }));
        return;
      }
      let intake = null;
      if (file) {
        const command = text.replace(/[\s。，,！!？?]/g, "");
        const resumesPreparedCase = Boolean(active.preparedIntake)
          && /^(?:开始分析|开始|继续|继续分析|按上文|按照上文|按上文记忆|按照上文记忆|执行|重新提交|再次提交|确认)$/.test(command);
        if (resumesPreparedCase) {
          intake = active.preparedIntake;
          updateActive((item) => ({ ...item, intakeState: "READY", updatedAt: now() }));
        } else {
          const rememberedNarrative = active.intakeDraft || recoverIntakeNarrative(active.messages);
          const accumulatedNarrative = [rememberedNarrative, text].filter(Boolean).join("\n补充信息：");
          if (accumulatedNarrative.length < 10) throw new Error("附图时请同时描述患者姓名、门诊号、性别、年龄、病灶部位与症状变化。");
          const extracted = await request("/v1/agent/intake/extract", { method: "POST", body: JSON.stringify({ narrative: accumulatedNarrative, model }) }, token);
          if (!extracted.complete) {
            updateActive((item) => ({ ...item, intakeDraft: accumulatedNarrative, preparedIntake: null, intakeState: "COLLECTING", updatedAt: now() }));
            addMessage({ role: "assistant", content: `请补充：**${(extracted.missing_field_labels || extracted.missing_fields).join("、")}**` });
            setLiveRun(null); return;
          }
          intake = extracted.intake;
          updateActive((item) => ({ ...item, intakeDraft: accumulatedNarrative, preparedIntake: intake, intakeState: "READY", updatedAt: now() }));
        }
      }
      const selectedRare = rarePackages.find((item) => item.package_id === active.selectedRareEvidenceId);
      const memoryContext = reasoningMode !== "standard" && selectedRare ? `\n\n[已选择罕见病增强证据包 ${selectedRare.package_id}]\nTop-K：${(selectedRare.top_predictions || []).map((item) => `${item.label} ${(Number(item.score) * 100).toFixed(1)}%`).join("；")}\nRAG：${selectedRare.rag?.answer || "无"}\n请将其视为本轮明确引用的证据，不要与其他增强结果混用。` : "";
      const form = new FormData(); form.append("message", ((text || "创建病例并执行分析") + memoryContext).slice(0, 4800)); form.append("case_id", active.caseId || ""); form.append("model", model); form.append("detail_level", reasoningMode === "standard" ? "concise" : "detailed"); form.append("reasoning_mode", reasoningMode === "deep" ? "deep" : "standard"); form.append("conversation_id", active.id); form.append("history_json", JSON.stringify(history));
      if (file) { form.append("image", file); form.append("intake_json", JSON.stringify(intake)); }
      const initial = await request("/v1/agent/runs/dispatch", { method: "POST", body: form, signal: controller.signal }, token);
      const payload = await pollRun(initial.run_id, controller.signal);
      const artifacts = payload.result?.kind === "chat" ? payload.result.chat?.artifacts || [] : [];
      const reportId = payload.result?.kind === "report" ? payload.result.report?.report_id : null;
      addMessage({ role: "assistant", content: answerFrom(payload), run: payload, pendingAction: payload.pending_action, runId: payload.run_id, caseId: payload.case_id, reportId, artifacts });
      if (payload.case_id) updateActive((item) => ({ ...item, caseId: payload.case_id }));
      if (payload.status === "WAITING_CONFIRM") updateActive((item) => ({ ...item, intakeState: "WAITING_CONFIRM" }));
      if (file && payload.status !== "WAITING_CONFIRM") {
        setAttachment(null);
        setAttachmentKind("normal");
        updateActive((item) => ({ ...item, intakeDraft: "", preparedIntake: null, intakeState: payload.case_id ? "CREATED" : "IDLE" }));
      }
    } catch (err) {
      if (err.name !== "AbortError" && err.status !== 401) addMessage({ role: "assistant", operationError: true, content: `操作未完成：${friendlyError(err)}` });
    } finally { setLiveRun(null); abortRef.current = null; }
  }

  async function approve(message, confirmed) {
    setApproving(true);
    try {
      const payload = await request(`/v1/agent/runs/${message.runId}/${confirmed ? "confirm" : "reject"}`, { method: "POST", body: confirmed ? JSON.stringify({ action_id: message.pendingAction.action_id }) : undefined }, token);
      updateActive((item) => ({ ...item, caseId: payload.case_id || item.caseId, messages: item.messages.map((entry) => entry.id === message.id ? { ...entry, pendingAction: null } : entry) }));
      addMessage({ role: "assistant", content: confirmed ? answerFrom(payload) : "已取消该操作。", run: payload, caseId: payload.case_id, reportId: payload.result?.report?.report_id });
      if (confirmed && message.pendingAction.kind === "CREATE_ANALYZE_CASE") {
        setAttachment(null);
        updateActive((item) => ({ ...item, intakeDraft: "", preparedIntake: null, intakeState: "CREATED" }));
      } else if (!confirmed && message.pendingAction.kind === "CREATE_ANALYZE_CASE") {
        updateActive((item) => ({ ...item, intakeState: "READY" }));
      }
    } catch (err) { addMessage({ role: "assistant", operationError: true, content: `确认操作失败：${friendlyError(err)}` }); }
    finally { setApproving(false); }
  }

  if (checking) return <div className="app-loading"><BrandMark size={54} /><LoaderCircle className="spin" /></div>;
  if (!token || !user) return <Login onLogin={handleLogin} />;
  return <div className="app-shell">
    <Sidebar open={sidebarOpen} threads={threads} activeId={active?.id} activeView={view?.type} user={user} onToggle={() => setSidebarOpen((value) => !value)} onNew={newChat} onSelect={(id) => { setActiveId(id); setView(null); if (window.innerWidth <= 860) setSidebarOpen(false); }} onRename={renameThread} onDelete={deleteThread} onWorkspace={(type) => { setView({ type }); if (window.innerWidth <= 860) setSidebarOpen(false); }} onLogout={logout} />
    <button className={`sidebar-scrim ${sidebarOpen ? "visible" : ""}`} aria-label="关闭侧栏" onClick={() => setSidebarOpen(false)} /><section className="chat-stage">
      <header className="topbar" inert={Boolean(view)}><button className="mobile-menu" aria-label="打开侧栏" onClick={() => setSidebarOpen((value) => !value)}><Menu /></button><div><strong>{active?.title || "新对话"}</strong><span>{active?.caseId ? `关联病例 ${active.caseId}` : "工作台"}</span></div><button className="top-workspace" onClick={() => setView({ type: active?.caseId ? "case" : "cases", id: active?.caseId })}><Activity size={16} />{active?.caseId ? "病例工作台" : "查看病例"}</button></header>
      <div className="chat-scroll" inert={Boolean(view)}><div className="message-stream">{active?.messages?.length ? active.messages.map((message) => <Message key={message.id} message={message} token={token} onOpen={setView} onApprove={approve} approving={approving} rareEvidencePackages={rarePackages} onOpenEvidence={setDetailPkg} />) : <EmptyChat onPrompt={setInput} user={user} />}{liveRun && <ExecutionCard run={liveRun} onStop={() => abortRef.current?.abort()} />}<div ref={bottomRef} /></div></div>
      <Composer inactive={Boolean(view)} value={input} onChange={setInput} onSend={send} onDocument={uploadDocument} attachment={attachment} setAttachment={setAttachment} attachmentKind={attachmentKind} setAttachmentKind={setAttachmentKind} intakeState={active?.intakeState} busy={Boolean(liveRun?.running)} models={models} model={model} setModel={setModel} token={token} rareEvidence={rarePackages.find((item) => item.package_id === active?.selectedRareEvidenceId) || null} onRareRun={runRareInference} onRareUse={useRareEvidence} onRareDeep={(evidence) => { setReasoningMode("deep"); useRareEvidence(evidence); }} reasoningMode={reasoningMode} setReasoningMode={setReasoningMode} lockedEvidenceId={reasoningMode === "deep" ? active?.selectedRareEvidenceId : null} evidencePackages={rarePackages} />
      <WorkspaceDrawer view={view} token={token} onDocumentUpdated={documentUpdated} onClose={() => setView(null)} onOpen={setView} onRareUse={useRareEvidence} onRarePersist={persistRarePackage} onRareDeep={(evidence) => { setReasoningMode("deep"); useRareEvidence(evidence); }} evidencePackages={rarePackages} selectedRareEvidenceId={active?.selectedRareEvidenceId} onDeleteEvidence={deleteRareEvidence} />
    </section>
    {detailPkg && <EvidenceDetail evidence={detailPkg} onClose={() => setDetailPkg(null)} onUse={(item) => { useRareEvidence(item); setDetailPkg(null); }} onDeep={(item) => { setReasoningMode("deep"); useRareEvidence(item); setDetailPkg(null); }} onDelete={(item) => { deleteRareEvidence(item); setDetailPkg(null); }} />}
  </div>;
}
