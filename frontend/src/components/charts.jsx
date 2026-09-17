import React, { useMemo } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Panel } from './primitives.jsx'

const AXIS = { fill: '#7b8ba6', fontSize: 10 }
const GRID = '#1d2634'

const clock = (iso) =>
  iso ? new Date(iso).toLocaleTimeString([], { minute: '2-digit', second: '2-digit' }) : ''

/** Series entries are `[dataKey, colour, label, axis]`, where `axis` is
 *  optionally `'right'`.
 *
 *  A second axis matters more than it sounds. On one shared axis, RPM near 4750
 *  forces the scale to 0-6000, which squeezes CHT (~180-290) and EGT (~700) into
 *  the bottom eighth of the plot: a 40 C rise during an overheating fault moves
 *  the line by under one percent of the chart height and reads as flat. Same
 *  problem for an anomaly score of 0-1 plotted against health of 0-100.
 */
export function TrendChart({ title, subtitle, history, series, height = 190 }) {
  const data = useMemo(
    () =>
      history.map((snap) => ({
        t: clock(snap.telemetry?.timestamp),
        ...snap.telemetry,
        health: snap.digital_twin?.health_score,
        anomaly_score: snap.prediction?.anomaly_score,
      })),
    [history],
  )
  const usesRight = series.some(([, , , axis]) => axis === 'right')
  return (
    <Panel title={title} subtitle={subtitle} className="chart">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 4, right: usesRight ? 0 : 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="t" tick={AXIS} minTickGap={28} />
          <YAxis yAxisId="left" tick={AXIS} />
          {usesRight && (
            <YAxis yAxisId="right" orientation="right" tick={AXIS} />
          )}
          <Tooltip contentStyle={{ background: '#0d1420', border: '1px solid #223049' }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {series.map(([key, colour, label, axis]) => (
            <Line
              key={key}
              yAxisId={axis === 'right' && usesRight ? 'right' : 'left'}
              name={label ?? key}
              type="monotone"
              dataKey={key}
              stroke={colour}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Panel>
  )
}

export function SpectrumChart({ spectrum, indicators }) {
  const rotational = indicators?.rotational_hz
  const firing = indicators?.firing_hz
  const orders = rotational
    ? [
        [0.5 * rotational, '0.5x misfire', '#fb6376'],
        [rotational, '1x imbalance', '#ffb25b'],
        [firing, '2x firing', '#40c9ff'],
      ]
    : []
  return (
    <Panel
      title="Vibration spectrum"
      subtitle={
        rotational
          ? `order tracked at ${rotational.toFixed(1)} Hz shaft speed - computed on the edge node`
          : 'awaiting data'
      }
      className="chart"
    >
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={spectrum} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="hz" tick={AXIS} unit="Hz" minTickGap={26} />
          <YAxis tick={AXIS} unit="g" />
          <Tooltip contentStyle={{ background: '#0d1420', border: '1px solid #223049' }} />
          {orders.map(([hz, label, colour]) => (
            <ReferenceLine
              key={label}
              x={Math.round(hz)}
              stroke={colour}
              strokeDasharray="3 3"
              label={{ value: label, fill: colour, fontSize: 9, position: 'top' }}
            />
          ))}
          <Bar dataKey="amplitude" fill="#3d7fd1" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

export function CylinderChart({ measured, expected, hottest, coldest }) {
  const data = (measured?.egt ?? []).map((egt, index) => ({
    name: `Cyl ${index + 1}`,
    egt,
    expected: expected?.egt?.[index],
    cht: measured?.cht?.[index],
    flag: index + 1 === hottest ? 'hot' : index + 1 === coldest ? 'cold' : '',
  }))
  return (
    <Panel
      title="Per-cylinder EGT"
      subtitle="a cold cylinder indicates misfire, a hot one indicates a lean injector"
      className="chart"
    >
      <ResponsiveContainer width="100%" height={186}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="name" tick={AXIS} />
          <YAxis tick={AXIS} unit="C" />
          <Tooltip contentStyle={{ background: '#0d1420', border: '1px solid #223049' }} />
          {data[0]?.expected != null && (
            <ReferenceLine
              y={data[0].expected}
              stroke="#6de6a9"
              strokeDasharray="4 4"
              label={{ value: 'expected', fill: '#6de6a9', fontSize: 9, position: 'right' }}
            />
          )}
          <Bar dataKey="egt" isAnimationActive={false}>
            {data.map((row) => (
              <Cell
                key={row.name}
                fill={row.flag === 'hot' ? '#fb6376' : row.flag === 'cold' ? '#40c9ff' : '#3d7fd1'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}

// A subsystem with no fitted trend reports hours-remaining-to-TBO as a
// placeholder. Drawn in the same colour as a real estimate, that makes an
// unmeasured subsystem look like a healthy one, so those bars are dimmed and the
// count is named in the subtitle.
const TREND_ESTABLISHED = 'trend_established'

export function RulChart({ rul }) {
  const subsystems = rul?.subsystems ?? {}
  const data = Object.entries(subsystems).map(([name, value]) => ({
    name: name.replace('_', ' '),
    hours: value.rul_hours,
    low: value.confidence_interval_hours?.[0],
    high: value.confidence_interval_hours?.[1],
    fitted: value.status === TREND_ESTABLISHED,
  }))
  const unfitted = data.filter((row) => !row.fitted).length
  return (
    <Panel
      title="Remaining useful life by subsystem"
      subtitle={
        `limiting: ${rul?.limiting_subsystem ?? '--'}` +
        (unfitted ? ` - ${unfitted} subsystem(s) dimmed: no fitted trend, showing TBO` : '')
      }
      className="chart"
    >
      <ResponsiveContainer width="100%" height={186}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 42, bottom: 0 }}>
          <CartesianGrid stroke={GRID} horizontal={false} />
          <XAxis type="number" tick={AXIS} unit="h" />
          <YAxis type="category" dataKey="name" tick={AXIS} width={82} />
          <Tooltip contentStyle={{ background: '#0d1420', border: '1px solid #223049' }} />
          <Bar dataKey="hours" fill="#6de6a9" isAnimationActive={false}>
            {data.map((row) => (
              <Cell
                key={row.name}
                fill={
                  !row.fitted
                    ? '#2c3a4f'
                    : row.hours < 50
                      ? '#fb6376'
                      : row.hours < 250
                        ? '#ffb25b'
                        : '#6de6a9'
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Panel>
  )
}
