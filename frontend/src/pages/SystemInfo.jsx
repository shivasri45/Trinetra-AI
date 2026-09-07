import React, { useEffect, useState } from 'react'
import { api } from '../services/api.js'
import { Badge, Empty, Panel, pretty } from '../components/primitives.jsx'

export default function SystemInfo({ snapshot, ingestion }) {
  const [metrics, setMetrics] = useState(null)
  const [icd, setIcd] = useState(null)
  const [posture, setPosture] = useState(null)
  const [frames, setFrames] = useState([])
  const [verified, setVerified] = useState(null)

  useEffect(() => {
    Promise.all([api.modelMetrics(), api.interfaceControl(), api.securityPosture()])
      .then(([m, i, p]) => {
        setMetrics(m)
        setIcd(i)
        setPosture(p)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (ingestion !== 'can') return setFrames([])
    const poll = () => api.frames().then((data) => setFrames(data.frames ?? [])).catch(() => {})
    poll()
    const timer = setInterval(poll, 2000)
    return () => clearInterval(timer)
  }, [ingestion])

  const verify = async () => {
    if (!snapshot) return
    try {
      const result = await api.verify(snapshot.telemetry, snapshot.integrity.signature)
      setVerified(result.valid ? 'signature valid' : 'signature INVALID')
    } catch (exc) {
      setVerified(exc.message)
    }
  }

  const tamper = async () => {
    if (!snapshot) return
    const modified = { ...snapshot.telemetry, rpm: 1 }
    const result = await api.verify(modified, snapshot.integrity.signature)
    setVerified(result.valid ? 'tampering NOT detected' : 'tampering correctly detected')
  }

  return (
    <>
      <Panel
        title="Deployed model provenance"
        subtitle="held-out performance of the models actually serving predictions"
      >
        {!metrics?.available && <Empty>No trained model found. Run the training command.</Empty>}
        {metrics?.available && (
          <>
            <div className="report">
              <div>
                <small>Accuracy</small>
                <strong>{metrics.classifier.accuracy}</strong>
              </div>
              <div>
                <small>Macro F1</small>
                <strong>{metrics.classifier.macro_f1}</strong>
              </div>
              <div>
                <small>Samples</small>
                <strong>{metrics.samples}</strong>
              </div>
              <div>
                <small>Features</small>
                <strong>{metrics.features}</strong>
              </div>
              <div>
                <small>Test sessions</small>
                <strong>{metrics.split.test_sessions}</strong>
              </div>
              <div>
                <small>Novelty false alarm</small>
                <strong>{metrics.novelty.healthy_false_alarm_rate}</strong>
              </div>
            </div>
            <p className="note">Split strategy: {metrics.split.strategy}.</p>
            <div className="factors">
              {Object.entries(metrics.classifier.per_class_f1).map(([name, score]) => (
                <div key={name}>
                  <span>{pretty(name)}</span>
                  <i className="bar">
                    <b style={{ width: `${score * 100}%` }} />
                  </i>
                  <em>{score.toFixed(3)}</em>
                </div>
              ))}
            </div>
            <h4>Most influential features</h4>
            <div className="factors compact">
              {metrics.top_features.slice(0, 10).map((item) => (
                <div key={item.feature}>
                  <span>{item.feature}</span>
                  <i className="bar">
                    <b style={{ width: `${(item.importance / metrics.top_features[0].importance) * 100}%` }} />
                  </i>
                  <em>{item.importance.toFixed(4)}</em>
                </div>
              ))}
            </div>
          </>
        )}
      </Panel>

      <Panel
        title="Security posture"
        subtitle="authentication, telemetry integrity and audit"
        actions={
          <div className="replay-controls">
            <button className="ghost" onClick={verify}>
              Verify current frame
            </button>
            <button className="ghost" onClick={tamper}>
              Simulate tampering
            </button>
          </div>
        }
      >
        {posture && (
          <div className="report">
            {Object.entries(posture).map(([key, value]) => (
              <div key={key}>
                <small>{pretty(key)}</small>
                <strong>{String(value)}</strong>
              </div>
            ))}
          </div>
        )}
        {snapshot?.integrity && (
          <p className="note">
            Current frame signature ({snapshot.integrity.algorithm}, key {snapshot.integrity.key_id}):{' '}
            <code>{snapshot.integrity.signature.slice(0, 32)}…</code>
          </p>
        )}
        {verified && <Badge tone={verified.includes('valid') || verified.includes('correctly') ? 'ok' : 'bad'}>{verified}</Badge>}
      </Panel>

      <Panel
        title="Engine data bus interface control"
        subtitle={
          icd
            ? `${icd.bus.protocol} at ${icd.bus.bitrate} bit/s - ${icd.bus.frames_per_sample} frames (${icd.bus.bytes_per_sample} bytes) per sample`
            : ''
        }
      >
        <p className="note">{icd?.note}</p>
        <table className="residuals">
          <thead>
            <tr>
              <th>Message</th>
              <th>PGN</th>
              <th>CAN ID</th>
              <th>Signals</th>
            </tr>
          </thead>
          <tbody>
            {(icd?.messages ?? []).map((message) => (
              <tr key={message.message}>
                <td>{message.message}</td>
                <td>{message.pgn}</td>
                <td>{message.can_id}</td>
                <td className="signals">
                  {message.signals
                    .map((s) => `${s.signal} @${s.start_byte}+${s.length_bytes}B ×${s.resolution}`)
                    .join(', ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Panel
        title="Live CAN traffic"
        subtitle={
          ingestion === 'can'
            ? 'frames from the most recent acquisition cycle'
            : 'switch ingestion to CAN on the toolbar to capture frames'
        }
      >
        {!frames.length && <Empty>No frames captured.</Empty>}
        {!!frames.length && (
          <div className="frames">
            {frames.map((frame, index) => (
              <div key={`${frame.message}-${index}`}>
                <b>{frame.can_id}</b>
                <span>{frame.message}</span>
                <code>{frame.data}</code>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </>
  )
}
