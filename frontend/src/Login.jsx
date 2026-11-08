import React, { useState } from "react";
import { ArrowRight, BookOpen, BrainCircuit, Eye, EyeOff, GitBranch, Layers3, LoaderCircle, ScanLine, ShieldCheck } from "lucide-react";
import { authApi } from "./api";
import { BrandLockup, BrandMark } from "./Brand";
import "./brand.css";

function EvidenceLandscape() {
  const modules = [
    { id: "brain", icon: BrainCircuit, title: "LLM 大脑", text: "理解问题 · 规划与改写查询" },
    { id: "vision", icon: ScanLine, title: "专业皮肤多模态模型", text: "联合阅读影像与病史" },
    { id: "embedding", icon: Layers3, title: "医学嵌入模型", text: "医学语料预训练 · 语义匹配" },
    { id: "tools", icon: GitBranch, title: "MCP 工具", text: "连接病例、报告与工具" },
    { id: "knowledge", icon: BookOpen, title: "RAG 领域知识库", text: "108.2万知识片段 · 十大疾病领域" },
  ];
  return <div className="xf-login-landscape xf-anatomy-stage" aria-label="信肤机器人：头脑规划问题，眼部阅读皮肤影像，胸腔匹配医学语义，手臂连接工具，并检索领域知识库">
    <div className="xf-anatomy-grid" />
    <svg className="xf-anatomy-connectors" viewBox="0 0 760 470" fill="none" preserveAspectRatio="none" aria-hidden="true">
      <ellipse cx="380" cy="420" rx="114" ry="20" fill="#061f1e" fillOpacity=".35" />
      <ellipse cx="380" cy="420" rx="150" ry="33" stroke="#a0c3ac" strokeOpacity=".1" />
      <path d="M207 60h74l59 14h18" /><path d="M551 125h-63l-61-18" />
      <path d="M207 208h79l57 15h20" /><path d="M551 295h-47l-34-11" />
      <path d="M207 354h65l53-60h26" />
      <g className="xf-anatomy-connector-signals"><circle cx="358" cy="74" r="3" /><circle cx="427" cy="107" r="3" /><circle cx="363" cy="223" r="3" /><circle cx="470" cy="284" r="3" /><circle cx="351" cy="294" r="3" /></g>
    </svg>
    <svg className="xf-anatomy-robot" viewBox="0 0 220 370" fill="none" role="img" aria-label="信肤机器人形象">
      <defs>
        <linearGradient id="xf-robot-shell" x1="51" y1="22" x2="174" y2="326" gradientUnits="userSpaceOnUse"><stop stopColor="#e5eee3" /><stop offset=".45" stopColor="#a9c8b7" /><stop offset="1" stopColor="#71998a" /></linearGradient>
        <linearGradient id="xf-robot-side" x1="39" y1="113" x2="181" y2="303" gradientUnits="userSpaceOnUse"><stop stopColor="#9fbfaa" /><stop offset=".6" stopColor="#4f8070" /><stop offset="1" stopColor="#a2bea8" /></linearGradient>
        <linearGradient id="xf-robot-face" x1="71" y1="58" x2="149" y2="91" gradientUnits="userSpaceOnUse"><stop stopColor="#173d39" /><stop offset=".6" stopColor="#244d43" /><stop offset="1" stopColor="#102f2e" /></linearGradient>
        <linearGradient id="xf-robot-core" x1="87" y1="164" x2="133" y2="213" gradientUnits="userSpaceOnUse"><stop stopColor="#365f4c" /><stop offset="1" stopColor="#153a32" /></linearGradient>
        <radialGradient id="xf-robot-halo"><stop stopColor="#b8d5a6" stopOpacity=".15" /><stop offset="1" stopColor="#b8d5a6" stopOpacity="0" /></radialGradient>
      </defs>
      <ellipse cx="110" cy="183" rx="104" ry="148" fill="url(#xf-robot-halo)" />
      <g className="xf-anatomy-robot-figure">
        <path d="M110 26V13" stroke="#9dbd9e" strokeWidth="3" strokeLinecap="round" />
        <circle cx="110" cy="10" r="5" fill="#d1dcb4" /><circle cx="110" cy="10" r="9" stroke="#bdd5ae" strokeOpacity=".15" />
        <rect x="58" y="28" width="104" height="89" rx="30" fill="url(#xf-robot-shell)" stroke="#d3e6d0" strokeOpacity=".7" />
        <path d="M72 38c7-5 15-7 27-7" stroke="#f2f6e9" strokeOpacity=".7" strokeWidth="2" strokeLinecap="round" />
        <path d="M54 64h-4v26h5M166 64h4v26h-5" stroke="#719a86" strokeWidth="5" strokeLinecap="round" />
        <rect x="72" y="59" width="76" height="35" rx="13" fill="url(#xf-robot-face)" stroke="#6b9985" />
        <path className="xf-anatomy-eyes" d="M88 74h9m26 0h9" stroke="#d3edcb" strokeWidth="5" strokeLinecap="round" />
        <path d="M104 86h12" stroke="#77a58b" strokeWidth="1.7" strokeLinecap="round" />
        <path d="M91 45h38M100 40v10m10-10v10m10-10v10" stroke="#728f72" strokeWidth="1.5" strokeLinecap="round" />
        <path d="M97 119h26v16H97z" fill="#6e9380" /><path d="M100 123h20" stroke="#b7d0b6" strokeWidth="2" />
        <circle cx="51" cy="150" r="17" fill="url(#xf-robot-side)" stroke="#a5c7af" strokeOpacity=".5" />
        <circle cx="169" cy="150" r="17" fill="url(#xf-robot-side)" stroke="#a5c7af" strokeOpacity=".5" />
        <rect x="29" y="159" width="25" height="74" rx="12" transform="rotate(5 29 159)" fill="url(#xf-robot-side)" stroke="#accbb1" strokeOpacity=".4" />
        <rect x="166" y="159" width="25" height="74" rx="12" transform="rotate(-5 166 159)" fill="url(#xf-robot-side)" stroke="#accbb1" strokeOpacity=".4" />
        <circle cx="38" cy="229" r="10" fill="#244f40" stroke="#aac5a9" strokeWidth="3" />
        <circle cx="182" cy="229" r="10" fill="#244f40" stroke="#aac5a9" strokeWidth="3" />
        <path d="M29 242h20v34c0 9-8 13-13 9l-12-10 5-33Z" fill="url(#xf-robot-shell)" stroke="#b3cbb0" strokeOpacity=".5" />
        <path d="M171 242h20l5 33-12 10c-5 4-13 0-13-9v-34Z" fill="url(#xf-robot-shell)" stroke="#b3cbb0" strokeOpacity=".5" />
        <path d="M28 266h10v11m154-11h-10v11" stroke="#47725e" strokeWidth="2" strokeLinecap="round" />
        <path d="M69 137c22-8 60-8 82 0 8 3 12 14 11 24l-7 72c-1 12-13 20-25 20H90c-12 0-24-8-25-20l-7-72c-1-10 3-21 11-24Z" fill="url(#xf-robot-shell)" stroke="#d0dfc7" strokeOpacity=".65" />
        <path d="M72 146c23-6 53-6 76 0" stroke="#eef3e0" strokeOpacity=".65" strokeWidth="1.5" />
        <rect x="81" y="161" width="58" height="57" rx="17" fill="url(#xf-robot-core)" stroke="#8fb98e" strokeOpacity=".7" />
        <path d="m96 178 14-8 14 8-14 8-14-8Z" fill="#d1e3bd" /><path d="m96 186 14 8 14-8m-28 8 14 8 14-8" stroke="#a7c99b" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="143" cy="172" r="2.5" fill="#e0ce95" /><path d="M78 230h64" stroke="#5c8d73" strokeWidth="2" strokeLinecap="round" /><path d="M87 237h46" stroke="#5c8d73" strokeWidth="1.4" strokeLinecap="round" />
        <path d="M78 256h64v19H78z" fill="#59836b" /><path d="M81 263h58" stroke="#b0c8a8" strokeWidth="2" />
        <rect x="73" y="276" width="31" height="56" rx="12" fill="url(#xf-robot-shell)" stroke="#b5cdb0" strokeOpacity=".55" />
        <rect x="116" y="276" width="31" height="56" rx="12" fill="url(#xf-robot-shell)" stroke="#b5cdb0" strokeOpacity=".55" />
        <path d="M76 324h24l4 15c1 7-4 12-11 12H67c-6 0-9-5-6-10l15-17Zm68 0h-24l-4 15c-1 7 4 12 11 12h26c6 0 9-5 6-10l-15-17Z" fill="#a9c3a6" stroke="#d0dfc3" strokeOpacity=".45" />
        <path d="M67 345h27m32 0h27" stroke="#4e775f" strokeWidth="2" strokeLinecap="round" />
      </g>
    </svg>
    <div className="xf-anatomy-cards">{modules.map(({ id, icon: Icon, title, text }) => <div key={id} className={`xf-anatomy-card xf-anatomy-card--${id}`}>
      <span className="xf-anatomy-card-icon"><Icon size={17} /></span><div><strong>{title}</strong><small>{text}</small></div>
    </div>)}</div>
  </div>;
}

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [visible, setVisible] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setError(""); setLoading(true);
    try { await onLogin(await authApi.login(username.trim(), password)); }
    catch (err) { setError(err.status === 401 ? "账号或密码不正确" : err.status === 429 ? "尝试次数较多，请稍后再试。" : "暂时无法登录，请稍后重试。"); }
    finally { setLoading(false); }
  }

  return <main className="xf-login xf-login--anatomy">
    <section className="xf-login-story">
      <div className="xf-login-ambient xf-login-ambient--one" /><div className="xf-login-ambient xf-login-ambient--two" />
      <header className="xf-login-story-head"><BrandLockup light /></header>
      <div className="xf-login-story-main">
        <h1>专注每一份<em>病例。</em></h1>
        <p className="xf-login-intro">整理记录，查找资料，衔接下一步工作。</p>
        <EvidenceLandscape />
      </div>
      <footer className="xf-login-story-foot"><span>病例与资料，在一处整理。</span><div><span><Layers3 size={13} />资料整理</span><span><ShieldCheck size={13} />病例管理</span></div></footer>
    </section>
    <section className="xf-login-access">
      <div className="xf-login-access-top"><span>信肤 · 工作台</span></div>
      <form className="xf-login-form" onSubmit={submit}>
        <div className="xf-login-form-glyph"><BrandMark size={48} /></div>
        <h2>欢迎回到工作台。</h2>
        <p className="xf-login-form-intro">输入账号，继续工作。</p>
        <label className="xf-login-label" htmlFor="xf-login-username">账号<input id="xf-login-username" autoFocus autoComplete="username" placeholder="输入账号" value={username} onChange={(event) => setUsername(event.target.value)} disabled={loading} /></label>
        <label className="xf-login-label" htmlFor="xf-login-password">密码<span className="xf-login-password-wrap"><input id="xf-login-password" type={visible ? "text" : "password"} autoComplete="current-password" placeholder="输入密码" value={password} onChange={(event) => setPassword(event.target.value)} disabled={loading} /><button type="button" className="xf-login-reveal" aria-label={visible ? "隐藏密码" : "显示密码"} aria-pressed={visible} onClick={() => setVisible(!visible)}>{visible ? <EyeOff size={16} /> : <Eye size={16} />}</button></span></label>
        {error && <div className="xf-login-error" role="alert">{error}</div>}
        <button className="xf-login-submit" disabled={loading || !username.trim() || !password}>{loading ? <LoaderCircle className="spin" size={17} /> : <span>进入工作台</span>}{loading ? "正在验证账号" : <ArrowRight size={17} />}</button>
      </form>
      <footer className="xf-login-access-foot"><span>信肤工作台</span><span>XINFU</span></footer>
    </section>
  </main>;
}
