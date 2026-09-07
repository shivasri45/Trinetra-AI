import React, { useCallback, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Cpu, History, Play, Radio, RotateCcw, Satellite, Square } from 'lucide-react'
import { api, telemetrySocket } from './services/api.js'
import Dashboard from './pages/Dashboard.jsx'
import Replay from './pages/Replay.jsx'
import SystemInfo from './pages/SystemInfo.jsx'
import { Badge, pretty } from './components/primitives.jsx'
import './styles.css'

const HISTORY_LENGTH = 120

const TABS = [
  ['dashboard', 'Live monitoring', Radio],
  ['replay', 'Mission replay', History],
  ['system', 'Models and interfaces', Cpu],
]

function App() {
  const [tab, setTab] = useState('dashboard')
  const [snapshot, setSnapshot] = useState(null)
  const [history, setHistory] = useState([])
  const [spectrum, setSpectrum] = useState([])
  const [alarms, setAlarms] = useState([])
  const [capabilities, setCapabilities] = useState(null)
  const [status, setStatus] = useState('connecting')
  const [running, setRunning] = useState(false)
  const [mission, setMission] = useState('normal_cruise')
  const [fault, setFault] = useState('normal')
  const [ingestion, setIngestion] = useState('direct')
  const [error, setError] = useState(null)

  useEffect(() => {
    api.capabilities().then(setCapabilities).catch((e) => setError(e.message))
    api.health()
      .then((h) => {
        setRunning(h.simulation_running)
        setIngestion(h.ingestion_mode)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    return telemetrySocket((next) => {
      setSnapshot(next)
      setHistory((current) => [...current.slice(-(HISTORY_LENGTH - 1)), next])
      setIngestion(next.ingestion?.source ?? 'direct')
      setFault(next.simulated_truth?.fault ?? 'normal')
    }, setStatus)
  }, [])

  useEffect(() => {
    if (tab !== 'dashboard' || !snapshot) return
    let active = true
    api.spectrum().then((data) => active && setSpectrum(data.spectrum)).catch(() => {})
    api.alarms().then((data) => active && setAlarms(data)).catch(() => {})
    return () => {
      active = false
    }
  }, [tab, snapshot?.sequence])

  const guard = useCallback(async (action) => {
    try {
      await action()
      setError(null)
    } catch (exc) {
      setError(exc.message)
    }
  }, [])

  const health = snapshot?.digital_twin?.operating_state ?? 'NORMAL'

  return (
    <main>
      <header>
        <div className="brand">
          <Satellite />
          <div>
            TRINETRA <b>AI</b>
            <small>AERO PISTON ENGINE DIGITAL TWIN</small>
          </div>
        </div>
        <nav>
          {TABS.map(([key, label, Icon]) => (
            <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>
              <Icon size={14} /> {label}
            </button>
          ))}
        </nav>
        <div className="status">
          <Badge tone={status === 'live' ? 'ok' : 'bad'}>
            <Radio size={11} /> {status}
          </Badge>
          <Badge>{ingestion === 'can' ? 'CAN BUS' : 'DIRECT'}</Badge>
          <span className={`badge state ${health}`}>{health}</span>
        </div>
      </header>

      <div className="toolbar">
        <div>
          <button
            onClick={() => guard(async () => {
              await api.start()
              setRunning(true)
            })}
            disabled={running}
          >
            <Play size={14} /> Start engine
          </button>
          <button
            className="ghost"
            onClick={() => guard(async () => {
              await api.stop()
              setRunning(false)
            })}
            disabled={!running}
          >
            <Square size={14} /> Stop and file mission
          </button>
          <button className="ghost" onClick={() => guard(async () => {
            await api.reset()
            setRunning(false)
            setHistory([])
          })}>
            <RotateCcw size={14} /> Reset
          </button>
        </div>

        <label>
          Mission profile
          <select
            value={mission}
            onChange={(e) => {
              setMission(e.target.value)
              guard(() => api.setMission(e.target.value))
            }}
          >
            {(capabilities?.mission_profiles ?? []).map((value) => (
              <option key={value} value={value}>
                {pretty(value)}
              </option>
            ))}
          </select>
        </label>

        <label>
          Fault injection
          <select
            value={fault}
            onChange={(e) => {
              setFault(e.target.value)
              guard(() => api.injectFault(e.target.value))
            }}
          >
            {(capabilities?.faults ?? []).map((value) => (
              <option key={value} value={value}>
                {pretty(value)}
              </option>
            ))}
          </select>
        </label>

        <label>
          Ingestion
          <select
            value={ingestion}
            onChange={(e) => {
              setIngestion(e.target.value)
              guard(() => api.setIngestion(e.target.value))
            }}
          >
            <option value="direct">Direct (in-process)</option>
            <option value="can">CAN bus (J1939-style)</option>
          </select>
        </label>
      </div>

      {error && <div className="error-bar">{error}</div>}

      {tab === 'dashboard' && (
        <Dashboard snapshot={snapshot} history={history} spectrum={spectrum} alarms={alarms} />
      )}
      {tab === 'replay' && <Replay />}
      {tab === 'system' && <SystemInfo snapshot={snapshot} ingestion={ingestion} />}

      <footer>
        {capabilities?.engine?.designation} - {capabilities?.engine?.displacement_l} L,{' '}
        {capabilities?.engine?.cylinders} cylinder, {capabilities?.engine?.rated_power_kw} kW rated,{' '}
        {capabilities?.engine?.tbo_hours} h TBO. Research prototype on synthetic telemetry. Not
        certified for flight, maintenance or airworthiness decisions.
      </footer>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<App />)
