import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play, SkipBack, Trash2 } from 'lucide-react'
import { api } from '../services/api.js'
import { CylinderChart, RulChart, TrendChart } from '../components/charts.jsx'
import { Diagnosis, HealthSummary, ResidualTable, SubsystemHealth } from '../components/panels.jsx'
import { Badge, Empty, Panel, pretty, stateClass } from '../components/primitives.jsx'

const SPEEDS = [0.5, 1, 2, 5, 10]
const RESIDUAL_CHANNELS = ['rpm', 'cht', 'egt', 'oil_pressure', 'oil_temperature', 'fuel_flow']

export default function Replay() {
  const [missions, setMissions] = useState([])
  const [selected, setSelected] = useState(null)
  const [snapshots, setSnapshots] = useState([])
  const [report, setReport] = useState(null)
  const [cursor, setCursor] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  const [error, setError] = useState(null)
  const timer = useRef(null)

  const loadMissions = useCallback(async () => {
    try {
      const data = await api.missions()
      setMissions(data.missions)
      if (data.missions.length && selected == null) setSelected(data.missions[0].id)
    } catch (exc) {
      setError(exc.message)
    }
  }, [selected])

  useEffect(() => {
    loadMissions()
  }, [])

  useEffect(() => {
    if (selected == null) return
    setPlaying(false)
    setCursor(0)
    setSnapshots([])
    Promise.all([api.missionSnapshots(selected, 0, 900), api.missionReport(selected)])
      .then(([data, summary]) => {
        setSnapshots(data.snapshots)
        setReport(summary)
        setError(null)
      })
      .catch((exc) => setError(exc.message))
  }, [selected])

  useEffect(() => {
    clearInterval(timer.current)
    if (!playing || !snapshots.length) return
    timer.current = setInterval(() => {
      setCursor((current) => {
        if (current >= snapshots.length - 1) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 1000 / speed)
    return () => clearInterval(timer.current)
  }, [playing, speed, snapshots.length])

  const current = snapshots[cursor]
  const window = useMemo(
    () => snapshots.slice(Math.max(0, cursor - 89), cursor + 1),
    [snapshots, cursor],
  )

  const remove = async (id) => {
    try {
      await api.deleteMission(id)
      setSelected(null)
      await loadMissions()
    } catch (exc) {
      setError(exc.message)
    }
  }

  return (
    <>
      <Panel
        title="Mission replay"
        subtitle="post-flight analysis from stored mission recordings"
        actions={
          <div className="replay-controls">
            <select value={selected ?? ''} onChange={(e) => setSelected(Number(e.target.value))}>
              {missions.map((mission) => (
                <option key={mission.id} value={mission.id}>
                  #{mission.id} - {pretty(mission.profile)} - {mission.samples} samples
                </option>
              ))}
            </select>
            <button onClick={() => setPlaying((p) => !p)} disabled={!snapshots.length}>
              {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? 'Pause' : 'Play'}
            </button>
            <button className="ghost" onClick={() => setCursor(0)} disabled={!snapshots.length}>
              <SkipBack size={14} /> Restart
            </button>
            <label>
              Speed
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
                {SPEEDS.map((value) => (
                  <option key={value} value={value}>
                    {value}x
                  </option>
                ))}
              </select>
            </label>
            {selected != null && (
              <button className="ghost danger" onClick={() => remove(selected)}>
                <Trash2 size={14} /> Delete
              </button>
            )}
          </div>
        }
      >
        {error && <Empty>{error}</Empty>}
        {!missions.length && !error && (
          <Empty>No recorded missions yet. Start the engine on the dashboard to record one.</Empty>
        )}
        {snapshots.length > 0 && (
          <div className="scrubber">
            <input
              type="range"
              min={0}
              max={snapshots.length - 1}
              value={cursor}
              onChange={(e) => setCursor(Number(e.target.value))}
            />
            <span>
              sample {cursor + 1} / {snapshots.length}
              {current?.telemetry?.engine_hours != null &&
                ` - ${current.telemetry.engine_hours.toFixed(1)} engine hours`}
            </span>
          </div>
        )}
      </Panel>

      {report?.summary && (
        <Panel title="Mission health report" subtitle={`derived from ${report.source} record`}>
          <div className="report">
            {[
              ['Samples', report.summary.samples],
              ['Health at start', report.summary.health_start],
              ['Health at end', report.summary.health_end],
              ['Lowest health', report.summary.health_min],
              ['Mean health', report.summary.health_mean],
              ['Anomalous fraction', report.summary.anomaly_sample_fraction],
              ['Engine hours', report.summary.engine_hours_end],
              ['Limiting subsystem', pretty(report.summary.final_rul?.limiting_subsystem)],
              ['RUL at end', report.summary.final_rul?.hours && `${report.summary.final_rul.hours} h`],
            ]
              .filter(([, value]) => value != null)
              .map(([label, value]) => (
                <div key={label}>
                  <small>{label}</small>
                  <strong className={label.includes('health') ? stateClass(Number(value)) : ''}>
                    {value}
                  </strong>
                </div>
              ))}
          </div>
          {report.summary.peak_readings && (
            <div className="report">
              {Object.entries(report.summary.peak_readings).map(([key, value]) => (
                <div key={key}>
                  <small>{pretty(key)}</small>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          )}
          {!!report.summary.alarms?.length && (
            <div className="alarms">
              {report.summary.alarms.map((entry) => (
                <div key={entry.sequence}>
                  <b>{entry.engine_hours?.toFixed?.(1)} h</b>
                  <span>
                    {pretty(entry.from)} → <strong>{pretty(entry.to)}</strong>
                  </span>
                  <small>health {entry.health}</small>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}

      {current && (
        <>
          <div className="replay-banner">
            <Badge tone="warn">REPLAY</Badge>
            <span>
              Mission #{selected} - {pretty(current.mission_profile)} - injected during recording:{' '}
              <strong>{pretty(current.simulated_truth?.fault ?? 'normal')}</strong>
            </span>
          </div>
          <HealthSummary snapshot={current} />
          <div className="grid">
            <div className="column">
              <TrendChart
                title="Rotational and thermal"
                subtitle="replayed telemetry - RPM left, temperatures right"
                history={window}
                series={[
                  ['rpm', '#40c9ff', 'RPM'],
                  ['cht', '#ffb25b', 'CHT °C', 'right'],
                  ['egt', '#fb6376', 'EGT °C', 'right'],
                ]}
              />
              <div className="pair">
                <CylinderChart
                  measured={current.cylinders}
                  expected={current.digital_twin?.cylinders?.expected}
                  hottest={current.telemetry?.hottest_cylinder}
                  coldest={current.telemetry?.coldest_cylinder}
                />
                <RulChart rul={current.rul} />
              </div>
            </div>
            <aside className="column">
              <Diagnosis snapshot={current} />
              <SubsystemHealth twin={current.digital_twin} />
              <ResidualTable twin={current.digital_twin} channels={RESIDUAL_CHANNELS} />
            </aside>
          </div>
        </>
      )}
    </>
  )
}
