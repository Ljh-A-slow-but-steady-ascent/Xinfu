import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, Check, Code2, Copy, ImageOff, LoaderCircle } from "lucide-react";
import { fetchBlob } from "./api";
import "./markdown.css";

const CITATION_PROTOCOL = "evidence-cite:";

/** Transform prose references without rewriting source code, links, or HTML. */
export function remarkEvidenceCitations() {
  return (tree) => {
    const numbers = new Map();
    function visit(parent) {
      if (!parent?.children || ["code", "inlineCode", "link", "linkReference", "html", "definition"].includes(parent.type)) return;
      parent.children = parent.children.flatMap((child) => {
        if (child.type !== "text") { visit(child); return [child]; }
        const value = String(child.value || "");
        const expression = /\[证据\s*[:：]\s*([^\]\n]+)\]/g;
        let match; let cursor = 0; const nodes = [];
        while ((match = expression.exec(value))) {
          if (match.index > cursor) nodes.push({ type: "text", value: value.slice(cursor, match.index) });
          const id = match[1].trim();
          if (!numbers.has(id)) numbers.set(id, numbers.size + 1);
          const number = numbers.get(id);
          nodes.push({ type: "link", url: `${CITATION_PROTOCOL}${number}:${encodeURIComponent(id)}`,
            children: [{ type: "text", value: `依据 ${number}` }] });
          cursor = expression.lastIndex;
        }
        if (!nodes.length) return [child];
        if (cursor < value.length) nodes.push({ type: "text", value: value.slice(cursor) });
        return nodes;
      });
    }
    visit(tree);
  };
}

export function safeUrlTransform(url) {
  const text = String(url || "");
  if (/^data:image\//i.test(text) || /^evidence-image:\/\//i.test(text)
      || text.startsWith(CITATION_PROTOCOL)) return text;
  return defaultUrlTransform(text);
}

export function SecureMarkdownImage({ src, alt, token, rarePackages, node: _node, ...props }) {
  const text = String(src || "");
  const isProtected = /^\/v1\/cases\/[^/]+\/image$/.test(text);
  const evidenceId = /^evidence-image:\/\/(.+)$/.exec(text)?.[1];
  const evidenceSrc = evidenceId
    ? (rarePackages || []).find((item) => item.package_id === evidenceId)?.image_preview || "" : "";
  const [resolved, setResolved] = useState({ source: "", url: "", failed: false });
  const [loadedSource, setLoadedSource] = useState("");
  const [failedSource, setFailedSource] = useState("");

  useEffect(() => {
    let active = true; let objectUrl = "";
    setResolved({ source: text, url: "", failed: false });
    if (!isProtected) return undefined;
    if (!token) { setResolved({ source: text, url: "", failed: true }); return undefined; }
    fetchBlob(text, token).then((blob) => {
      if (!active) return;
      objectUrl = URL.createObjectURL(blob);
      setResolved({ source: text, url: objectUrl, failed: false });
    }).catch(() => {
      if (active) setResolved({ source: text, url: "", failed: true });
    });
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [text, token, isProtected]);

  const imageSrc = evidenceId ? evidenceSrc : isProtected
    ? (resolved.source === text ? resolved.url : "") : text;
  const unavailable = !text || (evidenceId && !evidenceSrc)
    || (isProtected && resolved.source === text && resolved.failed)
    || (imageSrc && failedSource === imageSrc);
  if (unavailable) return <span className="md-image-state md-image-error"><ImageOff size={18} aria-hidden="true" />图片暂时无法显示</span>;
  if (isProtected && !imageSrc) return <span className="md-image-state" role="status"><LoaderCircle className="md-image-spinner" size={18} aria-hidden="true" />图片加载中</span>;
  return <span className={`md-image-frame ${loadedSource === imageSrc ? "is-loaded" : "is-loading"}`}>
    {loadedSource !== imageSrc && <span className="md-image-placeholder" aria-hidden="true" />}
    <img {...props} src={imageSrc} alt={alt || "图片"} loading="lazy"
      onLoad={() => { setLoadedSource(imageSrc); setFailedSource(""); }}
      onError={() => setFailedSource(imageSrc)} />
    {alt && <span className="md-image-caption">{alt}</span>}
  </span>;
}

function useCopy(text) {
  const [status, setStatus] = useState("idle");
  const reset = useRef(null);
  useEffect(() => () => clearTimeout(reset.current), []);
  async function copy() {
    clearTimeout(reset.current);
    try {
      if (!navigator.clipboard?.writeText) throw new Error("unavailable");
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch { setStatus("error"); }
    reset.current = setTimeout(() => setStatus("idle"), 2400);
  }
  return [status, copy];
}

function codeText(children) {
  return React.Children.toArray(children).map((child) => {
    if (typeof child === "string" || typeof child === "number") return String(child);
    return React.isValidElement(child) ? codeText(child.props.children) : "";
  }).join("");
}

function HighlightedCode({ text }) {
  // Every token remains a React text child, so source cannot introduce markup.
  const tokens = useMemo(() => {
    const expression = /(#.*$|\/\/.*$|\/\*[\s\S]*?\*\/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b(?:async|await|break|case|catch|class|const|continue|def|del|do|elif|else|except|export|false|False|finally|for|from|function|if|import|in|interface|let|None|null|of|pass|raise|return|switch|throw|true|True|try|type|var|while|with|yield)\b|\b\d+(?:\.\d+)?\b)/gm;
    const result = []; let match; let cursor = 0;
    while ((match = expression.exec(text))) {
      if (match.index > cursor) result.push(text.slice(cursor, match.index));
      const token = match[0];
      const kind = /^(#|\/\/|\/\*)/.test(token) ? "comment"
        : /^["'`]/.test(token) ? "string" : /^\d/.test(token) ? "number" : "keyword";
      result.push(<span className={`md-token-${kind}`} key={match.index}>{token}</span>);
      cursor = expression.lastIndex;
    }
    if (cursor < text.length) result.push(text.slice(cursor));
    return result;
  }, [text]);
  return tokens;
}

function CodeBlock({ children }) {
  const child = React.Children.toArray(children).find((item) => React.isValidElement(item));
  const language = /language-([\w+-]+)/.exec(child?.props?.className || "")?.[1] || "code";
  const text = codeText(children).replace(/\n$/, "");
  const [status, copy] = useCopy(text);
  const labels = { python: "Python", py: "Python", javascript: "JavaScript", js: "JavaScript", typescript: "TypeScript", ts: "TypeScript", jsx: "JSX", tsx: "TSX", json: "JSON", sql: "SQL", bash: "Shell", shell: "Shell", sh: "Shell", html: "HTML", css: "CSS", markdown: "Markdown", md: "Markdown", code: "代码" };
  return <div className="md-code-block">
    <div className="md-code-toolbar"><span><Code2 size={14} aria-hidden="true" />{labels[language.toLowerCase()] || language}</span>
      <button type="button" onClick={copy} className={status === "copied" ? "is-copied" : ""}
        title={status === "error" ? "复制失败，请选择代码后复制" : "复制代码"} aria-label="复制代码">
        {status === "copied" ? <Check size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}
        <span role="status" aria-live="polite">{status === "copied" ? "已复制" : status === "error" ? "复制失败" : "复制"}</span>
      </button>
    </div>
    <pre tabIndex={0}><code className={`language-${language}`}><HighlightedCode text={text} /></code></pre>
  </div>;
}

function MarkdownLink({ href, children, node: _node, ...props }) {
  if (String(href || "").startsWith(CITATION_PROTOCOL)) {
    const match = /^evidence-cite:(\d+):(.*)$/.exec(href);
    if (match) {
      let id;
      try { id = decodeURIComponent(match[2]); } catch { id = match[2]; }
      return <span className="md-citation" tabIndex={0} title={id} aria-label={`依据 ${match[1]}`}>
        <BookOpen size={11} aria-hidden="true" />依据 {match[1]}
      </span>;
    }
  }
  return <a {...props} href={href} target="_blank" rel="noreferrer noopener">{children}</a>;
}

export default function Markdown({ children, token, rarePackages }) {
  return <div className="rich-markdown"><ReactMarkdown urlTransform={safeUrlTransform}
    remarkPlugins={[remarkGfm, remarkEvidenceCitations]} components={{
      a: MarkdownLink,
      pre: CodeBlock,
      p: ({ children: content }) => <p className="md-paragraph">{content}</p>,
      h1: ({ children: content }) => <h1 className="md-heading">{content}</h1>,
      h2: ({ children: content }) => <h2 className="md-heading">{content}</h2>,
      h3: ({ children: content }) => <h3 className="md-heading">{content}</h3>,
      h4: ({ children: content }) => <h4 className="md-heading">{content}</h4>,
      code: ({ children: content, className, node: _node, ...props }) => <code {...props} className={className}>{content}</code>,
      table: ({ children: content }) => <div className="table-scroll md-table-scroll" tabIndex={0}><table>{content}</table></div>,
      img: (props) => <SecureMarkdownImage {...props} token={token} rarePackages={rarePackages} />,
      hr: () => <hr className="md-divider" />,
    }}>{String(children || "")}</ReactMarkdown></div>;
}
