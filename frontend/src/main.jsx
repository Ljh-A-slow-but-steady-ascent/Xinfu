import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./workspace.css";
import "./markdown.css";

class WorkspaceBoundary extends React.Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", background: "#f5f7f1", color: "#294b40", textAlign: "center" }}><div><h1>信肤<span style={{ color: "#ac9566" }}>.</span></h1><p>页面未能打开，请重试。</p><button onClick={() => location.reload()}>重新打开</button></div></main>;
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <WorkspaceBoundary><App /></WorkspaceBoundary>
  </React.StrictMode>,
);
