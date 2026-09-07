import React from 'react'

export const pretty = (value) =>
  value ? String(value).replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()) : value

export const stateClass = (health, warning = 80, critical = 50) =>
  health == null ? '' : health > warning ? 'ok' : health > critical ? 'warn' : 'bad'

export function Panel({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-head">
          <div>
            {title && <h3>{title}</h3>}
            {subtitle && <small>{subtitle}</small>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

export function Metric({ label, value, unit, tone = '', hint }) {
  return (
    <div className="metric" title={hint}>
      <span>{label}</span>
      <strong className={tone}>{value ?? '--'}</strong>
      {unit && <small>{unit}</small>}
    </div>
  )
}

export function Bar({ value, max = 100, tone = '' }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <i className="bar">
      <b className={tone} style={{ width: `${pct}%` }} />
    </i>
  )
}

export function Badge({ children, tone = '' }) {
  return <span className={`badge ${tone}`}>{children}</span>
}

export function Empty({ children }) {
  return <p className="empty">{children}</p>
}
