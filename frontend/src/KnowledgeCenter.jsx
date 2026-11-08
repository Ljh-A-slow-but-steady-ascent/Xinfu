import { Fragment, useEffect, useRef, useState } from "react";
import { Activity, BarChart3, BookOpen, BrainCircuit, Check, ChevronRight, CircleAlert, CircleDot, Database, FileText, GitBranch, Gauge, ImagePlus, Layers3, LoaderCircle, LockKeyhole, Microscope, Play, RefreshCw, ScanSearch, SlidersHorizontal, Square, Upload, X } from "lucide-react";
import { request, fetchBlob } from "./api";
import "./knowledge.css";
import "./knowledge-layout.css";

const RISK_LABELS = { routine: "常规", referral: "转诊", urgent: "优先", any: "待定", emergency: "急症" };
const GROUP_COLORS = { nevus: "#398a7d", melanoma: "#bc766a", other: "#6888a3" };

function StatCard({ label, value, caption, icon, tone = "teal" }) {
  return <div className={`kb-stat-card stat-${tone}`}><div className="kb-stat-icon">{icon}</div><span>{label}</span><strong>{value}</strong><small>{caption}</small></div>;
}

function ParameterBars({ layers = [] }) {
  const max = Math.max(1, ...layers.map((layer) => Number(layer.parameters || 0)));
  return <div className="parameter-bars">{layers.map((layer) => <div className="parameter-row" key={layer.id}><div className="parameter-row-head"><span>{layer.label}</span><b>{Number(layer.parameters || 0).toLocaleString()}</b></div><div className="parameter-track"><i className={layer.state === "TRAINABLE" ? "trainable" : "frozen"} style={{ width: `${Math.max(3, Number(layer.parameters || 0) / max * 100)}%` }} /></div><small><span className={layer.state === "TRAINABLE" ? "status-live" : "status-lock"}>{layer.state === "TRAINABLE" ? "可训练" : "冻结"}</span>{layer.role}</small></div>)}</div>;
}

function ModelTopology({ layers = [] }) {
  return <div className="topology-flow">{layers.map((layer, index) => <div className="topology-step" key={layer.id}><div className={`topology-node ${layer.state === "TRAINABLE" ? "node-trainable" : "node-frozen"}`}><span>{layer.state === "TRAINABLE" ? <SlidersHorizontal size={16} /> : <LockKeyhole size={16} />}</span><strong>{layer.label}</strong><small>{layer.state === "TRAINABLE" ? "可训练层" : "冻结层"}</small></div>{index < layers.length - 1 && <ChevronRight className="topology-arrow" size={16} />}</div>)}</div>;
}

function Histogram({ title, subtitle, bins = [], color = "teal", meta = [] }) {
  const max = Math.max(1, ...bins);
  return <div className="histogram-card"><div className="histogram-head"><div><strong>{title}</strong><small>{subtitle}</small></div><span className={`histogram-badge ${color}`}>DISTRIBUTION</span></div><div className="histogram-bars">{bins.map((value, index) => <i key={index} style={{ height: `${Math.max(8, value / max * 100)}%` }} className={color}><em /></i>)}</div><div className="histogram-axis"><span>低</span><span>均值附近</span><span>高</span></div>{meta.length > 0 && <div className="histogram-meta">{meta.map((item) => <span key={item.label}><small>{item.label}</small><b>{item.value}</b></span>)}</div>}</div>;
}

function EmbeddingMap({ points = [], summary = {} }) {
  const clusters = summary.clusters || [];
  return <div className="embedding-card"><div className="histogram-head"><div><strong>样本表示分布</strong><small>{summary.projection || "PCA-2D"} · {summary.sample_count || points.length} 个样本快照</small></div><span className="histogram-badge blue">EMBEDDING</span></div><svg viewBox="0 0 100 100" className="embedding-map" role="img" aria-label="模型样本表示分布"><path d="M8 92H94M8 92V7" className="embedding-axis" />{[25, 50, 75].map((n) => <g key={n}><line x1={n} x2={n} y1="8" y2="92" className="embedding-grid" /><line x1="8" x2="94" y1={100 - n} y2={100 - n} className="embedding-grid" /></g>)}{points.map((point, index) => <circle key={`${point.group}-${index}`} cx={point.x} cy={100 - point.y} r="2.5" fill={GROUP_COLORS[point.group] || GROUP_COLORS.other} className="embedding-point" />)}</svg><div className="embedding-legend">{clusters.length ? clusters.map((cluster) => <span key={cluster.group}><i style={{ background: GROUP_COLORS[cluster.group] || GROUP_COLORS.other }} />{cluster.label} · {cluster.count}</span>) : <><span><i style={{ background: GROUP_COLORS.nevus }} />色素痣</span><span><i style={{ background: GROUP_COLORS.melanoma }} />高风险候选</span><span><i style={{ background: GROUP_COLORS.other }} />其他候选</span></>}</div><div className="embedding-summary"><span><small>分离度</small><b>{summary.separation_score ? `${Math.round(summary.separation_score * 100)}%` : "—"}</b></span><span><small>簇数量</small><b>{summary.cluster_count || clusters.length || "—"}</b></span><span><small>投影</small><b>{summary.projection || "PCA-2D"}</b></span></div></div>;
}

function DistributionDeck({ items = [], type = "weights" }) {
  const isWeights = type === "weights";
  return <div className={`distribution-deck ${isWeights ? "weights-deck" : "activations-deck"}`}>{items.map((item, index) => <Histogram key={item.name || item.short_name || index} title={item.name || item.short_name} subtitle={isWeights ? `${item.shape || "候选张量"} · 梯度范数 ${Number(item.gradient_norm || 0).toFixed(3)}` : `激活张量 · 稀疏度 ${Math.round(Number(item.sparsity || 0) * 100)}%`} bins={item.bins || []} color={isWeights ? (index === 2 ? "blue" : "teal") : (index === 1 ? "violet" : "blue")} meta={isWeights ? [{ label: "均值", value: Number(item.mean || 0).toFixed(3) }, { label: "std", value: Number(item.std || 0).toFixed(3) }, { label: "更新", value: `+${(Number(item.delta || 0) * 100).toFixed(1)}%` }] : [{ label: "均值", value: Number(item.mean || 0).toFixed(2) }, { label: "std", value: Number(item.std || 0).toFixed(2) }, { label: "P95", value: Number(item.p95 || 0).toFixed(2) }]} />)}</div>;
}

function WeightTable({ rows = [] }) {
  return <div className="weight-table-wrap"><div className="weight-table-title"><div><strong>权重分布明细</strong><small>相对基线的统计摘要</small></div><span><CircleDot size={14} />{rows.length} 个张量</span></div><div className="weight-table-scroll"><table className="weight-table"><thead><tr><th>张量</th><th>均值</th><th>标准差</th><th>范围</th><th>变化</th></tr></thead><tbody>{rows.map((row) => <tr key={row.name}><td><b>{row.name}</b></td><td>{Number(row.mean).toFixed(3)}</td><td>{Number(row.std).toFixed(3)}</td><td>{Number(row.min).toFixed(2)} ~ {Number(row.max).toFixed(2)}</td><td><span className="delta-value">+{(Number(row.delta) * 100).toFixed(1)}%</span></td></tr>)}</tbody></table></div></div>;
}

function Overview({ data, onRefresh, loading, error }) {
  const sources = data?.sources || [];
  const domains = data?.domain_distribution || [];
  const total = data?.domain_total_chunks;
  const max = Math.max(1, ...domains.map((item) => item.chunks));
  const compact = (value) => `${Number(value / 1000).toLocaleString("en-US")}k`;
  return <div className="kb-overview">
    <div className="kb-hero knowledge-hero"><div><span className="kb-kicker"><BookOpen size={14} />领域知识库</span><h2>资料总览</h2><p>按疾病领域查看知识片段与资料分布。</p><div className="hero-tags"><span><Layers3 size={13} />{data?.domain_count ?? "—"} 个疾病领域</span><span>{total ? `${(total / 10000).toLocaleString("zh-CN")} 万 Chunk` : "正在读取"}</span></div></div><button className="kb-refresh" onClick={onRefresh} disabled={loading}><RefreshCw className={loading ? "spin" : ""} size={14} />{loading ? "刷新中" : "刷新"}</button><div className="knowledge-hero-art" aria-hidden="true"><i /><i /><i /><span><BookOpen size={28} /></span></div></div>
    {error && <div className="knowledge-state state-error" role="status"><CircleAlert size={17} /><span>{error}</span><button onClick={onRefresh}>重新加载</button></div>}
    <div className="kb-stat-grid">
      <StatCard label="知识片段" value={total ? `${(total / 10000).toLocaleString("zh-CN")}万` : "—"} caption={total ? `${total.toLocaleString("en-US")} Chunk` : "正在读取"} icon={<Layers3 />} />
      <StatCard label="疾病领域" value={data?.domain_count ?? "—"} caption="皮肤专科知识分类" icon={<BookOpen />} />
      <StatCard label="资料来源" value={data?.source_count ?? "—"} caption="指南与研究资料" icon={<GitBranch />} />
      <StatCard label="待审核" value={data?.pending_upload_count ?? "—"} caption="新上传内容" icon={<Upload />} tone="blue" />
    </div>
    <div className="knowledge-overview-grid knowledge-domain-grid">
      <section className="kb-panel knowledge-domain-table"><div className="kb-panel-title"><div><span>疾病领域</span><small>领域知识片段总览</small></div><BookOpen size={18} /></div>
        <table aria-label="疾病领域 Chunk 数量"><thead><tr><th>疾病领域</th><th>Chunk 数量</th></tr></thead><tbody>{domains.map((item, index) => <tr key={item.label}><td><span className="domain-rank">{String(index + 1).padStart(2, "0")}</span>{item.label}</td><td>{compact(item.chunks)}</td></tr>)}</tbody><tfoot><tr><td>总计</td><td>{total ? compact(total) : "—"}</td></tr></tfoot></table>
      </section>
      <section className="kb-panel knowledge-domain-distribution"><div className="kb-panel-title"><div><span>领域分布</span><small>各疾病领域的 Chunk 规模</small></div><Layers3 size={18} /></div>
        <div className="domain-distribution">{domains.map((item, index) => <div className="domain-distribution-row" key={item.label}><div><span>{item.label}</span><strong>{compact(item.chunks)}</strong></div><div className="domain-bar-track"><i style={{ width: `${item.chunks / max * 100}%`, animationDelay: `${index * 30}ms` }} /></div></div>)}</div>
      </section>
    </div>
    <div className="knowledge-source-footer"><span><GitBranch size={15} />资料来源</span><div>{sources.map((source) => <span key={source.short_name} title={source.name}>{source.short_name}</span>)}</div></div>
  </div>;
}

function KnowledgeItems({ token }) {
  const [items, setItems] = useState([]); const [query, setQuery] = useState(""); const [source, setSource] = useState(""); const [active, setActive] = useState(null);
  const [loading, setLoading] = useState(true); const [error, setError] = useState("");
  async function load() { setLoading(true); setError(""); const params = new URLSearchParams(); if (query) params.set("query", query); if (source) params.set("source", source); try { const value = await request(`/v1/knowledge/items?${params}`, {}, token); setItems(value.items || []); } catch { setError("内容暂时无法加载，请稍后重试。"); } finally { setLoading(false); } }
  useEffect(() => { load(); }, [token, source]);
  async function openItem(item) { try { setActive(await request(`/v1/knowledge/items/${encodeURIComponent(item.document_id)}`, {}, token)); } catch { setError("这份资料暂时无法打开。"); } }
  return <div className={`kb-items ${active ? "has-detail" : ""}`}><header className="knowledge-section-heading"><div><span>资料库</span><h2>内容</h2><p>按疾病、主题或来源查找资料。</p></div><span className="knowledge-count">{items.length} 份资料</span></header><div className="kb-list-toolbar"><div className="kb-search"><BookBookIcon /><input aria-label="搜索资料" value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === "Enter" && load()} placeholder="搜索疾病、主题或正文" /></div><select aria-label="资料来源" value={source} onChange={(event) => setSource(event.target.value)}><option value="">全部来源</option><option value="AAD">AAD</option><option value="DermNet">DermNet</option><option value="CDC">CDC</option><option value="NICE">NICE</option><option value="项目资料">项目资料</option></select><button className="kb-soft-button" onClick={load} disabled={loading}><RefreshCw size={14} className={loading ? "spin" : ""} />查找</button></div>{error && <div className="knowledge-state state-error" role="status"><CircleAlert size={17} /><span>{error}</span><button onClick={load}>重试</button></div>}<div className="knowledge-reading-layout"><div className="kb-item-list">{loading && !items.length ? <div className="knowledge-state"><LoaderCircle className="spin" size={20} /><span>正在加载内容</span></div> : items.length ? items.map((item) => <button key={item.document_id} className={`kb-item-row ${active?.document_id === item.document_id ? "selected" : ""}`} onClick={() => openItem(item)}><span className="kb-item-icon"><FileText size={17} /></span><span><strong>{item.title}</strong><small>{item.source} · {item.year} · {item.chunk_count} 个片段</small></span><em className={`risk-${item.risk}`}>{RISK_LABELS[item.risk] || item.risk}</em><ChevronRight size={16} /></button>) : <div className="knowledge-empty"><BookOpen size={28} /><h3>没有找到相关内容</h3><p>换一个关键词或来源试试。</p></div>}</div>{active && <div className="kb-item-detail"><div className="kb-detail-head"><div><span>{active.source} · {active.year}</span><h3>{active.title}</h3></div><button aria-label="关闭资料详情" onClick={() => setActive(null)}><X size={17} /></button></div><div className="kb-frontmatter">{[["来源", active.source_name], ["版本", active.front_matter?.version || "未标注"], ["内容", `${active.chunk_count} 个片段`]].map(([label, value]) => <span key={label}><b>{label}</b>{value}</span>)}</div><div className="kb-chunk-list">{(active.chunks || []).map((chunk, index) => <article key={chunk.id}><span><i>{String(index + 1).padStart(2, "0")}</i>{chunk.section}</span><p>{chunk.content}</p></article>)}</div>{/^https?:\/\//.test(active.source_url || "") && <a className="knowledge-original-link" href={active.source_url} target="_blank" rel="noreferrer">查看来源原文<ChevronRight size={14} /></a>}</div>}</div></div>;
}

function BookBookIcon() { return <BookOpen size={15} />; }

const PAD_MODE_LABEL = { image_only: "仅图像", text_only: "仅临床文本", image_text: "图像 + 文本融合" };

/* 按真实置信度推演样本掌握时序：高置信先掌握，低置信后掌握。
   墙的最终对错构成由当前模式的真实 accuracy 决定；缓动分布让翻转全程持续。 */
function buildMasteryWall(pool, acc) {
  if (!pool.length) return [];
  const nC = Math.max(10, Math.min(22, Math.round(24 * acc)));
  const correct = pool.filter((c) => c.correct).sort((a, b) => b.confidence - a.confidence);
  const wrong = pool.filter((c) => !c.correct);
  const wall = [];
  const useC = Math.min(nC, correct.length);
  correct.slice(0, useC).forEach((c, i) => {
    const frac = Math.pow((i + 1) / useC, 0.75);
    const e = Math.max(2, Math.min(28, 2 + Math.round(frac * 26)));
    wall.push({ ...c, learnEpoch: e });
  });
  wrong.slice(0, Math.max(0, 24 - useC)).forEach((c) => wall.push({ ...c, learnEpoch: Infinity }));
  return wall;
}

function CaseThumb({ token, imgId, alt }) {
  const [url, setUrl] = useState("");
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true; let objectUrl = ""; setUrl(""); setFailed(false);
    fetchBlob(`/v1/observatory/pad/image/${encodeURIComponent(imgId)}`, token).then((blob) => { if (alive) { objectUrl = URL.createObjectURL(blob); setUrl(objectUrl); } }).catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [imgId, token]);
  return url ? <img src={url} alt={alt || imgId} loading="lazy" /> : <div className="case-thumb-loading" aria-label={failed ? "图片暂不可用" : "正在加载图片"}>{failed ? <ImagePlus size={17} /> : <LoaderCircle size={16} className="spin" />}</div>;
}

function CurveChart({ points, upto }) {
  const [hover, setHover] = useState(null);
  if (!points || points.length < 2) return <div className="curve-empty">暂无曲线数据</div>;
  const W = 640, H = 260, L = 46, B = 26, T = 14, R = 46;
  const step = Math.max(1, Math.floor(points.length / 120));
  const sampled = points.filter((_, i) => i % step === 0 || i === points.length - 1);
  const pts = upto != null ? sampled.slice(0, Math.max(2, upto)) : sampled;
  const losses = sampled.flatMap((p) => [p.train_loss, p.val_loss]);
  const lMin = Math.min(...losses) * 0.9, lMax = Math.max(...losses) * 1.08;
  const accs = sampled.map((p) => p.val_accuracy);
  const aMin = Math.max(0, Math.min(...accs) - 0.12), aMax = 1;
  const x = (i) => L + (i / (pts.length - 1)) * (W - L - R);
  const yL = (v) => T + (1 - (v - lMin) / (lMax - lMin)) * (H - T - B);
  const yA = (v) => T + (1 - (v - aMin) / (aMax - aMin)) * (H - T - B);
  function smooth(getY, key) {
    const P = pts.map((p, i) => [x(i), getY(p[key])]);
    if (P.length < 3) return `M${P.map((q) => q.map((n) => n.toFixed(1)).join(",")).join("L")}`;
    let d = `M${P[0][0].toFixed(1)},${P[0][1].toFixed(1)}`;
    for (let i = 1; i < P.length - 1; i++) {
      const mx = (P[i][0] + P[i + 1][0]) / 2, my = (P[i][1] + P[i + 1][1]) / 2;
      d += ` Q${P[i][0].toFixed(1)},${P[i][1].toFixed(1)} ${mx.toFixed(1)},${my.toFixed(1)}`;
    }
    const lastP = P[P.length - 1];
    d += ` L${lastP[0].toFixed(1)},${lastP[1].toFixed(1)}`;
    return d;
  }
  return <div className="curve-wrap">
    <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={(e) => { const rect = e.currentTarget.getBoundingClientRect(); const px = ((e.clientX - rect.left) / rect.width) * W; const idx = Math.max(0, Math.min(pts.length - 1, Math.round(((px - L) / (W - L - R)) * (pts.length - 1)))); setHover(idx); }} onMouseLeave={() => setHover(null)}>
      {[0.2, 0.4, 0.6, 0.8, 1].map((t) => { const v = lMin + (lMax - lMin) * t; return <g key={t}><line x1={L} x2={W - R} y1={yL(v)} y2={yL(v)} className="curve-grid" /><text x={L - 6} y={yL(v) + 3.5} className="curve-label" textAnchor="end">{v.toFixed(1)}</text></g>; })}
      <path d={smooth(yL, "train_loss")} className="curve-line curve-train" />
      <path d={smooth(yL, "val_loss")} className="curve-line curve-val" />
      <path d={smooth(yA, "val_accuracy")} className="curve-line curve-acc" />
      <circle cx={x(pts.length - 1)} cy={yL(pts[pts.length - 1].val_loss)} r="3.5" className="curve-dot curve-val-dot" />
      <circle cx={x(pts.length - 1)} cy={yA(pts[pts.length - 1].val_accuracy)} r="3" className="curve-dot curve-acc-dot" />
      {hover != null && <g><line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} className="curve-cursor" /><circle cx={x(hover)} cy={yL(pts[hover].train_loss)} r="3.5" className="curve-dot curve-train-dot" /><circle cx={x(hover)} cy={yL(pts[hover].val_loss)} r="3.5" className="curve-dot curve-val-dot" /><circle cx={x(hover)} cy={yA(pts[hover].val_accuracy)} r="3" className="curve-dot curve-acc-dot" /></g>}
      <text x={W - R + 6} y={yA(aMax) + 4} className="curve-label">100%</text>
      <text x={W - R + 6} y={yA(aMin) + 4} className="curve-label">{(aMin * 100).toFixed(0)}%</text>
      <text x={L} y={H - 6} className="curve-label">epoch 1</text>
      <text x={W - R} y={H - 6} className="curve-label" textAnchor="end">epoch {pts[pts.length - 1].epoch}</text>
    </svg>
    <div className="curve-legend"><span className="lg lg-train">train loss</span><span className="lg lg-val">val loss</span><span className="lg lg-acc">val accuracy</span>{hover != null && <em>epoch {pts[hover].epoch} · train {pts[hover].train_loss.toFixed(3)} · val {pts[hover].val_loss.toFixed(3)} · acc {(pts[hover].val_accuracy * 100).toFixed(1)}%</em>}</div>
  </div>;
}

function ConfusionMatrix({ matrix, classes }) {
  if (!matrix || !matrix.length) return null;
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0)) || [];
  const max = Math.max(...matrix.flat()) || 1;
  return <div className="confusion-wrap"><div className="confusion-grid" style={{ gridTemplateColumns: `44px repeat(${classes.length}, minmax(0,1fr)) 24px` }}>
    <span />
    {classes.map((c) => <span key={c} className="confusion-axis">{c}</span>)}
    <span className="confusion-axis confusion-more">…</span>
    {matrix.map((row, r) => <Fragment key={r}>
      <span className="confusion-axis">{classes[r]}</span>
      {row.map((v, c) => <span key={c} className={`confusion-cell ${r === c ? "diag" : ""} ${v > 0 ? "hot" : ""}`} style={{ background: v > 0 ? `rgba(74,122,168,${0.12 + 0.78 * (v / max)})` : undefined, color: v / max > 0.55 ? "#fff" : undefined }} title={`真实 ${classes[r]} → 预测 ${classes[c]}：${v} 例${rowTotals[r] ? `（${((v / rowTotals[r]) * 100).toFixed(0)}%）` : ""}`}>{v || ""}</span>)}
      <span className="confusion-cell confusion-more-cell">…</span>
    </Fragment>)}
    <span className="confusion-axis confusion-more">…</span>
    {classes.map((c) => <span key={c} className="confusion-cell confusion-more-cell">…</span>)}
    <span className="confusion-cell confusion-more-cell">…</span>
  </div><div className="confusion-note">横轴 = 模型预测 · 纵轴 = 病理真实标签</div></div>;
}

function CaseExplorer({ token }) {
  const [mode, setMode] = useState("image_text");
  const [filter, setFilter] = useState("all");
  const [label, setLabel] = useState("all");
  const [data, setData] = useState(null);
  const [detail, setDetail] = useState(null);
  const detailRef = useRef(null);
  useEffect(() => { setData(null); request(`/v1/observatory/pad/cases?mode=${mode}&filter=${filter}&label=${label}&limit=36`, {}, token).then(setData).catch(() => setData({ cases: [], total: 0 })); }, [mode, filter, label, token]);
  useEffect(() => {
    const dialog = detailRef.current;
    if (!detail || !dialog) return;
    dialog.showModal();
    return () => dialog.close();
  }, [detail]);
  const CLASS = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"];
  return <div className="case-explorer">
    <div className="kb-panel-title"><div><span>样本复核</span><small>PAD-UFES-20 测试集 · 共 {data?.total ?? "…"} 条</small></div><ScanSearch size={18} /></div>
    <div className="case-filters">
      <div className="filter-group">{Object.entries(PAD_MODE_LABEL).map(([k, v]) => <button key={k} className={mode === k ? "active" : ""} onClick={() => setMode(k)}>{v}</button>)}</div>
      <div className="filter-group">{[["all", "全部"], ["correct", "预测正确"], ["incorrect", "预测错误"]].map(([k, v]) => <button key={k} className={filter === k ? "active" : ""} onClick={() => setFilter(k)}>{v}</button>)}</div>
      <div className="filter-group">{[["all", "全部类型"], ...CLASS.map((name) => [name, name])].map(([k, v]) => <button key={k} className={label === k ? "active" : ""} onClick={() => setLabel(k)}>{v}</button>)}</div>
    </div>
    {!data ? <div className="knowledge-state"><LoaderCircle size={18} className="spin" /><span>正在加载样本</span></div> : !data.cases.length ? <div className="knowledge-empty"><ScanSearch size={28} /><h3>当前筛选下暂无样本</h3><p>尝试切换模型、结果或类型。</p></div> : <div className="case-grid">{data.cases.map((item) => <button key={item.img_id} className={`case-card ${item.correct ? "ok" : "bad"}`} onClick={() => setDetail(item)}>
      <CaseThumb token={token} imgId={item.img_id} alt={item.true_cn} />
      <div className="case-card-meta"><strong>{item.true_cn}</strong><span className={item.correct ? "badge-ok" : "badge-bad"}>{item.correct ? "预测正确" : `误判为 ${item.pred_cn}`}</span><small>{(item.confidence * 100).toFixed(1)}%</small></div>
    </button>)}</div>}
    {detail && <dialog ref={detailRef} className="case-detail-mask" aria-label="样本详情" onCancel={(event) => { event.preventDefault(); setDetail(null); }} onClick={() => setDetail(null)}>
      <div className="case-detail" onClick={(e) => e.stopPropagation()}>
        <div className="case-detail-head"><strong>{detail.img_id}</strong><span className={detail.correct ? "badge-ok" : "badge-bad"}>{detail.correct ? "预测正确" : "预测错误"}</span><button aria-label="关闭样本详情" onClick={() => setDetail(null)}><X size={15} /></button></div>
        <div className="case-detail-body">
          <div className="case-detail-img"><CaseThumb token={token} imgId={detail.img_id} alt={detail.true_cn} /></div>
          <div className="case-detail-info">
            <div className="case-truepred"><div><small>病理真实</small><strong>{detail.true_cn}（{detail.true}）</strong></div><div><small>模型预测</small><strong className={detail.correct ? "badge-ok" : "badge-bad"}>{detail.pred_cn}（{detail.pred}）</strong></div></div>
            <div className="case-probbars">{detail.top3.map((p) => <div key={p.name} className="prob-row"><span>{p.cn} {p.name}</span><i style={{ width: `${Math.max(2, p.prob * 100)}%` }} /><em>{(p.prob * 100).toFixed(1)}%</em></div>)}</div>
            <div className="case-meta-grid">{[["年龄", detail.meta.age], ["性别", detail.meta.gender], ["部位", detail.meta.region], ["Fitzpatrick", detail.meta.fitspatrick], ["直径 mm", detail.meta.diameter], ["活检", detail.meta.biopsed === "True" ? "是" : detail.meta.biopsed === "False" ? "否" : ""]].map(([k, v]) => v ? <div key={k}><small>{k}</small><span>{v}</span></div> : null)}</div>
            {detail.prompt && <div className="case-prompt"><small>融合模型输入的临床文本</small><p>{detail.prompt}</p></div>}
          </div>
        </div>
      </div>
    </dialog>}
  </div>;
}

function TrainingObservatory({ token }) {
  const [data, setData] = useState(null);
  const [dataError, setDataError] = useState("");
  const [snapshot, setSnapshot] = useState(null); const [inspectView, setInspectView] = useState("weights");
  const [curveMode, setCurveMode] = useState("image_text");
  const [curves, setCurves] = useState(null);
  const [split, setSplit] = useState("test");
  const [play, setPlay] = useState(false);
  const [playEpoch, setPlayEpoch] = useState(0);
  const [streamCurrent, setStreamCurrent] = useState(null);
  const [wall, setWall] = useState([]);
  const [wallProgress, setWallProgress] = useState({ mastered: 0, errors: 0 });
  const [samplePool, setSamplePool] = useState([]); const fileRef = useRef(null); const [uploadNotice, setUploadNotice] = useState("");
  useEffect(() => { request("/v1/observatory/datasets", {}, token).then(setData).catch(() => { setData(null); setDataError("训练记录暂时无法加载。"); }); }, [token]);
  useEffect(() => { request("/v1/observatory/model-snapshot", {}, token).then((value) => setSnapshot(value || null)).catch(() => setSnapshot(null)); }, [token]);
  useEffect(() => { setCurves(null); request(`/v1/observatory/pad/curves?mode=${curveMode}`, {}, token).then(setCurves).catch(() => setCurves(null)); }, [curveMode, token]);
  useEffect(() => { setPlay(false); setWall([]); setStreamCurrent(null); setSamplePool([]); setWallProgress({ mastered: 0, errors: 0 }); request(`/v1/observatory/pad/cases?mode=${curveMode}&limit=48`, {}, token).then((d) => setSamplePool(d.cases || [])).catch(() => setSamplePool([])); }, [curveMode, token]);
  useEffect(() => {
    if (!play) return;
    const pts = curves?.points || [];
    const total = pts.length;
    const pool = samplePool;
    if (!total || !pool.length) return;
    const acc = (data?.pad?.modes || []).find((m) => m.mode === curveMode)?.[split]?.accuracy ?? 0.8;
    const w = buildMasteryWall(pool, acc);
    if (!w.length) return;
    setWall(w);
    let e = 0;
    setPlayEpoch(0); setWallProgress({ mastered: 0, errors: w.length });
    let finishTimer; const timer = setInterval(() => {
      e += 1;
      setPlayEpoch(e);
      const mastered = w.filter((s) => s.learnEpoch <= e).length;
      setWallProgress({ mastered, errors: w.length - mastered });
      const flip = w.find((s) => s.learnEpoch === e);
      setStreamCurrent(flip || w[(e * 5) % w.length]);
      if (e >= total) { clearInterval(timer); finishTimer = setTimeout(() => setPlay(false), 1400); }
    }, 260);
    return () => { clearInterval(timer); clearTimeout(finishTimer); };
  }, [play, curves, data, curveMode, split, samplePool]);
  async function uploadDataset(file) { if (!file) return; const form = new FormData(); form.append("file", file); try { const result = await request("/v1/knowledge/uploads", { method: "POST", body: form }, token); setUploadNotice(`已接收 ${result.filename}，待审核`); } catch (err) { setUploadNotice("未能上传，请重试。"); } }
  const pad = data?.pad;
  const modes = pad?.modes || [];
  const activeMode = modes.find((m) => m.mode === curveMode) || modes[0];
  const splitData = activeMode ? activeMode[split] : null;
  const classNames = pad?.class_names || [];
  const f1Entries = Object.entries(splitData?.per_class_f1 || {}).sort((a, b) => a[0].localeCompare(b[0]));
  const pts = curves?.points || [];
  const livePoint = pts.length ? pts[Math.max(0, (play ? playEpoch : pts.length) - 1)] : null;
  return <div className="kb-training">
    <div className="kb-hero training-hero"><div><span className="kb-kicker"><Activity size={14} />训练记录</span><h2>训练观察</h2><p>回看训练过程，比较分类表现，复核验证样本。</p><div className="hero-tags"><span><Microscope size={13} />PAD-UFES-20{pad?.image_count ? ` · ${pad.image_count.toLocaleString()} 例` : ""}</span><span><Layers3 size={13} />图像与临床文本</span></div></div></div>
    {dataError && <div className="knowledge-state state-error" role="status"><CircleAlert size={17} /><span>{dataError}</span></div>}
    <div className="training-metrics"><StatCard label="准确率" value={splitData ? `${(splitData.accuracy * 100).toFixed(1)}%` : "—"} caption={`${PAD_MODE_LABEL[curveMode]} · ${split === "test" ? "测试集" : "验证集"}`} icon={<Activity />} /><StatCard label="均衡准确率" value={splitData ? `${(splitData.balanced_accuracy * 100).toFixed(1)}%` : "—"} caption="Balanced Accuracy" icon={<Gauge />} tone="blue" /><StatCard label="Macro-F1" value={splitData ? `${(splitData.macro_f1 * 100).toFixed(1)}%` : "—"} caption="各类别 F1 平均值" icon={<BarChart3 />} tone="violet" /></div>
    <div className="mode-chips">{modes.map((m) => <button key={m.mode} className={`mode-chip ${curveMode === m.mode ? "active" : ""}`} onClick={() => setCurveMode(m.mode)}>{PAD_MODE_LABEL[m.mode]}<em>{((m[split]?.accuracy || 0) * 100).toFixed(1)}%</em></button>)}<div className="filter-group split-toggle">{[["test", "测试集"], ["validation", "验证集"]].map(([k, v]) => <button key={k} className={split === k ? "active" : ""} onClick={() => setSplit(k)}>{v}</button>)}</div></div>
    <div className="kb-panel stage-panel">
      <div className="stage-grid">
        <div className="stage-curve">
          <div className="stage-head"><strong>{PAD_MODE_LABEL[curveMode]}</strong><button className="play-button" onClick={() => setPlay((value) => !value)} disabled={!pts.length || !samplePool.length}>{play ? <Square size={13} /> : <Play size={14} />}{play ? "停止回放" : "回放训练"}</button></div>
          <div className="big-metric">
            <strong>{livePoint ? `${(livePoint.val_accuracy * 100).toFixed(1)}%` : "…"}</strong>
            <div className="big-metric-side"><span>验证准确率</span><em>第 {livePoint ? (play ? playEpoch : pts.length) : "—"} / {pts.length || "—"} 轮</em>{livePoint && <em>验证损失 {livePoint.val_loss.toFixed(3)} · 训练损失 {livePoint.train_loss.toFixed(3)}</em>}</div>
          </div>
          <CurveChart points={pts} upto={play ? playEpoch : null} />
        </div>
        <div className="stage-right">
          <div className="knowledge-sample-heading"><strong>样本回放</strong><span className="knowledge-example-badge">示例</span></div>
          {streamCurrent && <div className={`sample-stream stage-stream ${play ? "live" : ""}`}>
            <div className="stream-thumb">{streamCurrent && <CaseThumb token={token} imgId={streamCurrent.img_id} alt="" />}</div>
            <div className="stream-info">
              {streamCurrent && <span className={streamCurrent.learnEpoch <= (play ? playEpoch : pts.length) ? "badge-ok" : "badge-bad"}>{streamCurrent.learnEpoch <= (play ? playEpoch : pts.length) ? "预测正确" : `误判 ${streamCurrent.pred_cn}`}</span>}
              <p>{streamCurrent?.prompt || "…"}</p>
              <div className="stream-counters">
                <div className="sc sc-ok"><strong>{wallProgress.mastered}</strong><small>复核通过</small></div>
                <div className="sc-arrow">→</div>
                <div className="sc sc-bad"><strong>{wallProgress.errors}</strong><small>错误</small></div>
              </div>
            </div>
          </div>}
          {!streamCurrent && <div className="stream-placeholder"><Play size={20} /><p>点击回放，查看样本变化</p></div>}
          <div className="wall-caption">验证样本 · {wall.length || Math.min(24, samplePool.length)} 例</div>
          <div className="mastery-wall">{wall.map((s) => <span key={s.img_id} className={`wall-thumb ${s.learnEpoch === Infinity ? "never" : s.learnEpoch <= (play ? playEpoch : pts.length) ? "ok" : "pending"}`} title={`${s.true_cn}${s.correct ? "" : " → 误判为 " + s.pred_cn}`}><CaseThumb token={token} imgId={s.img_id} alt="" /></span>)}</div>
        </div>
      </div>
    </div>
    <div className="training-layout layout-half">
      <div className="kb-panel"><div className="kb-panel-title"><div><span>混淆矩阵</span><small>{split === "test" ? "测试" : "验证"}集</small></div><BrainCircuit size={18} /></div><ConfusionMatrix matrix={splitData?.confusion_matrix} classes={classNames} /></div>
      <div className="kb-panel"><div className="kb-panel-title"><div><span>逐类别 F1</span><small>{split === "test" ? "测试" : "验证"}集</small></div><BarChart3 size={18} /></div>
        {f1Entries.length ? <div className="f1-bars">{f1Entries.map(([name, value]) => <div key={name} className="f1-row"><span>{pad?.class_names_cn?.[name] || name}<small>{name}</small></span><i style={{ width: `${value * 100}%` }} /><em>{(value * 100).toFixed(1)}</em></div>)}</div> : <div className="knowledge-state"><BarChart3 size={20} /><span>暂无逐类数据</span></div>}
      </div>
    </div>
    {snapshot && <section className="kb-panel knowledge-inspector"><div className="kb-panel-title"><div><span>结构与表示</span><small>权重、激活与样本表示</small></div><span className="knowledge-example-badge">示例</span></div><div className="knowledge-inspector-tabs" role="tablist" aria-label="观察内容">{[["weights", "权重"], ["activations", "激活"], ["embeddings", "样本表示"], ["structure", "结构"]].map(([key, label]) => <button role="tab" aria-selected={inspectView === key} key={key} className={inspectView === key ? "active" : ""} onClick={() => setInspectView(key)}>{label}</button>)}</div><div className="knowledge-inspector-body" key={inspectView}>{inspectView === "weights" && <><DistributionDeck items={snapshot.weight_histograms || []} /><WeightTable rows={snapshot.weight_histograms || []} /></>}{inspectView === "activations" && <DistributionDeck items={snapshot.activation_distributions || []} type="activations" />}{inspectView === "embeddings" && <EmbeddingMap points={snapshot.embedding_map || []} summary={snapshot.embedding_summary || {}} />}{inspectView === "structure" && <><ModelTopology layers={snapshot.layers || []} /><ParameterBars layers={snapshot.layers || []} /></>}</div></section>}
    <CaseExplorer token={token} />
    <div className="kb-panel training-input"><div className="kb-panel-title"><div><span>数据集上传</span><small>上传新样本</small></div><Upload size={18} /></div><label className="kb-upload-box"><input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp,.md,.txt,.json,.csv" hidden onChange={(event) => { uploadDataset(event.target.files?.[0]); event.target.value = ""; }} /><ImagePlus size={23} /><strong>选择图片或文本数据</strong><small>JPG / PNG / WebP / MD / TXT / CSV</small></label>{uploadNotice && <div className="input-trace"><span><CircleDot size={12} />{uploadNotice}</span></div>}</div>
  </div>;
}

export default function KnowledgeCenter({ token }) {
  const [tab, setTab] = useState("overview"); const [overview, setOverview] = useState(null); const uploadRef = useRef(null); const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(true); const [error, setError] = useState(""); const [uploading, setUploading] = useState(false);
  async function load() { setLoading(true); setError(""); try { setOverview(await request("/v1/knowledge/overview", {}, token)); } catch { setError("资料暂时无法加载。"); } finally { setLoading(false); } }
  useEffect(() => { load(); }, [token]);
  async function upload(file) { if (!file) return; setUploading(true); const form = new FormData(); form.append("file", file); try { const result = await request("/v1/knowledge/uploads", { method: "POST", body: form }, token); setNotice(`已接收 ${result.filename}，待审核`); await load(); } finally { setUploading(false); } }
  return <div className="kb-center"><div className="kb-tabs" role="tablist" aria-label="资料与训练"><div className="tabs-brand"><BookOpen size={16} /><span>资料</span></div><button role="tab" aria-selected={tab === "overview"} className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")}><Database size={15} />资料总览</button><button role="tab" aria-selected={tab === "items"} className={tab === "items" ? "active" : ""} onClick={() => setTab("items")}><BookOpen size={15} />内容</button><button role="tab" aria-selected={tab === "training"} className={tab === "training" ? "active" : ""} onClick={() => setTab("training")}><Activity size={15} />训练观察</button><input ref={uploadRef} type="file" accept=".md,.txt,.json,.csv,image/jpeg,image/png,image/webp" hidden onChange={(event) => { upload(event.target.files?.[0]).catch(() => setNotice("上传失败，请稍后重试")); event.target.value = ""; }} /><button className="kb-upload-button" disabled={uploading} onClick={() => uploadRef.current?.click()}>{uploading ? <LoaderCircle className="spin" size={14} /> : <Upload size={14} />}{uploading ? "正在上传" : "上传资料"}</button></div>{notice && <div className="kb-notice" role="status"><Check size={15} />{notice}<button aria-label="关闭提示" onClick={() => setNotice("")}><X size={13} /></button></div>}<div className="knowledge-tab-panel" key={tab} role="tabpanel">{tab === "overview" && <Overview data={overview} onRefresh={load} loading={loading} error={error} />}{tab === "items" && <KnowledgeItems token={token} />}{tab === "training" && <TrainingObservatory token={token} />}</div></div>;
}
