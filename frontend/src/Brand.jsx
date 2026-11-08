import React from "react";
import "./brand.css";

/** Layered skin / connected evidence. Kept as vectors for crisp small UI sizes. */
export function BrandMark({ size = 32, className = "", tone = "default", ...props }) {
  return <span className={`xf-brand-mark xf-brand-mark--${tone} ${className}`} style={{ width: size, height: size }} {...props}>
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <path d="m9.5 18 14.5-8 14.5 8L24 26 9.5 18Z" fill="currentColor" />
      <path d="m10 25 14 7.75L38 25M10 32l14 7.75L38 32" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="38" cy="10" r="3.2" className="xf-brand-signal" />
    </svg>
  </span>;
}

export function BrandLockup({ compact = false, light = false, className = "" }) {
  return <div className={`xf-brand-lockup ${light ? "xf-brand-lockup--light" : ""} ${className}`}>
    <BrandMark size={compact ? 32 : 42} tone={light ? "light" : "default"} />
    <span className="xf-brand-wordmark"><strong>信肤<span className="xf-brand-wordmark-dot">.</span></strong>{!compact && <small>XINFU</small>}</span>
  </div>;
}
