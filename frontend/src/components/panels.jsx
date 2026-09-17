import React from 'react'
import { Activity, AlertTriangle, Gauge, ShieldCheck, Wrench } from 'lucide-react'
import { Bar, Badge, Empty, Metric, Panel, pretty, stateClass } from './primitives.jsx'

// RUL is only a number when a degradation trend actually supports one. Before
// MIN_SAMPLES the estimator falls back to hours-remaining-to-TBO, and with a flat
// trend it does the same - both are placeholders, not predictions. Rendering them
// as a figure puts the most optimistic possible value next to a critical health
// index, which reads as the panels contradicting each other. Show the state
// instead, and keep the number for the case where a trend was really fitted.
function rulDisplay(rul) {
  if (!rul) return { value: '--', unit: '', tone: '' }

  // Not enough samples yet: the estimator has nothing to fit, so there is no
  // number to show. Brief, and honest about being brief.
  if (rul.status === 'establishing_trend') {
    return { value: 'Establishing', unit: 'collecting samples for a trend', tone: '' }
  }

  // No measurable degradation. This is a result, not a gap: life is limited by
  // the published overhaul interval rather than by observed wear, so the hours
  // are real and worth showing - just qualified so they are not read as a
  // degradation-based prediction.
  if (rul.status === 'no_degradation_trend') {
    return { value: `${rul.hours} h`, unit: 'TBO limited - no measurable wear', tone: 'ok' }
  }

  const ci = rul.confidence_interval_hours
  return {
    value: `${rul.hours} h`,
    unit: ci ? `95% CI ${ci[0]}-${ci[1]} h` : '',
    tone: rul.hours < 50 ? 'bad' : rul.hours < 250 ? 'warn' : 'ok',
  }
}

export function HealthSummary({ snapshot }) {
  const twin = snapshot?.digital_twin
  const prediction = snapshot?.prediction
  const rul = snapshot?.rul
  const health = twin?.health_score
  const tone = stateClass(health)
  const rulView = rulDisplay(rul)
  return (
    <section className="health">
      <div className={`health-ring ${tone}`}>
        <span>{health ?? '--'}</span>
        <small>HEALTH INDEX</small>
      </div>
      <Metric
        label="DIAGNOSIS"
        value={pretty(prediction?.predicted_fault)}
        unit={prediction ? `${prediction.confidence}% confidence` : ''}
        tone={prediction?.anomaly ? 'bad' : 'ok'}
      />
      <Metric
        label="REMAINING USEFUL LIFE"
        value={rulView.value}
        unit={rulView.unit}
        tone={rulView.tone}
        hint={rul?.method}
      />
      <Metric
        label="ANOMALY SCORE"
        value={prediction?.anomaly_score}
        unit={prediction?.anomaly ? 'deviation confirmed' : 'within noise'}
        tone={prediction?.anomaly ? 'warn' : ''}
      />
      <Metric
        label="ENGINE HOURS"
        value={snapshot?.telemetry?.engine_hours}
        unit={`of ${rul?.tbo_hours ?? '--'} h TBO`}
      />
    </section>
  )
}

export function SubsystemHealth({ twin }) {
  const subsystems = twin?.subsystems ?? {}
  return (
    <Panel title="Subsystem health" subtitle="acute residuals, learned drift and condition indicators">
      <div className="subsystems">
        {Object.entries(subsystems).map(([name, value]) => (
          <div key={name} className="subsystem">
            <span>{pretty(name)}</span>
            <Bar value={value.health} tone={stateClass(value.health)} />
            <strong className={stateClass(value.health)}>{value.health}</strong>
            <small>
              acute {value.acute_penalty} / drift {value.chronic_penalty} / indicator{' '}
              {value.indicator_penalty}
            </small>
          </div>
        ))}
        {!Object.keys(subsystems).length && <Empty>Awaiting telemetry.</Empty>}
      </div>
    </Panel>
  )
}

export function Diagnosis({ snapshot }) {
  const prediction = snapshot?.prediction
  const explanation = snapshot?.explanation
  if (!prediction) return <Panel title="Diagnosis"><Empty>Awaiting telemetry.</Empty></Panel>
  const anomalous = prediction.anomaly
  return (
    <Panel
      title={
        <>
          <Activity size={15} /> Diagnosis and evidence
        </>
      }
      subtitle={`${explanation?.attribution_method ?? ''} - detector backend: ${
        prediction.model_backend
      }`}
      className="insight"
    >
      <h2 className={anomalous ? 'bad' : 'ok'}>
        {anomalous ? <AlertTriangle size={18} /> : <ShieldCheck size={18} />}{' '}
        {pretty(prediction.predicted_fault)}
      </h2>
      <p>{explanation?.summary}</p>
      <p className="mechanism">{explanation?.mechanism}</p>

      <div className="factors">
        {(explanation?.evidence ?? []).map((item) => (
          <div key={item.feature}>
            <span>{item.feature}</span>
            <Bar value={item.contribution} />
            <em>{item.contribution}%</em>
            <small>
              {item.channel
                ? `${item.measured} vs ${item.expected} ${item.unit ?? ''} (${item.residual_sigma}σ ${
                    item.direction
                  })`
                : item.detail}
            </small>
          </div>
        ))}
      </div>

      <div className="detectors">
        {Object.entries(prediction.detectors ?? {}).map(([name, detail]) => (
          <div key={name}>
            <small>{pretty(name)}</small>
            <b>
              {name === 'statistical'
                ? `mean NIS ${detail.mean_nis} / limit ${detail.threshold}`
                : name === 'unsupervised'
                  ? `${detail.novelty_raw ?? '--'} / limit ${detail.threshold ?? '--'}`
                  : `${detail.top_class} p=${detail.probability}`}
            </b>
          </div>
        ))}
      </div>

      <footer className="agreement">
        <Badge tone={explanation?.model_agreement?.state === 'agree' ? 'ok' : 'warn'}>
          model vs physics: {explanation?.model_agreement?.state ?? '--'}
        </Badge>
        <Badge tone={explanation?.sensor_plausibility === 'CONSISTENT' ? 'ok' : 'warn'}>
          sensors: {explanation?.sensor_plausibility?.toLowerCase()}
        </Badge>
        {prediction.detection_triggers?.map((trigger) => (
          <Badge key={trigger}>{pretty(trigger)}</Badge>
        ))}
      </footer>
    </Panel>
  )
}

export function ResidualTable({ twin, channels }) {
  const normalised = twin?.normalised_residuals ?? {}
  const expected = twin?.expected ?? {}
  const measured = twin?.measured ?? {}
  const estimated = twin?.estimated ?? {}
  return (
    <Panel
      title="Twin synchronisation"
      subtitle={`Kalman innovations in sigma - mean NIS ${twin?.mean_nis ?? '--'}`}
    >
      <table className="residuals">
        <thead>
          <tr>
            <th>Channel</th>
            <th>Measured</th>
            <th>Estimated</th>
            <th>Expected</th>
            <th>Innovation</th>
          </tr>
        </thead>
        <tbody>
          {channels.map((channel) => {
            const sigma = normalised[channel]
            const tone = Math.abs(sigma ?? 0) > 3 ? 'bad' : Math.abs(sigma ?? 0) > 1.5 ? 'warn' : ''
            return (
              <tr key={channel}>
                <td>{pretty(channel)}</td>
                <td>{measured[channel]?.toFixed?.(1) ?? '--'}</td>
                <td>{estimated[channel]?.toFixed?.(1) ?? '--'}</td>
                <td>{expected[channel]?.toFixed?.(1) ?? '--'}</td>
                <td className={tone}>{sigma?.toFixed?.(2) ?? '--'} σ</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </Panel>
  )
}

export function TelemetryGrid({ telemetry, fields }) {
  return (
    <Panel title="Live sensor telemetry" subtitle="as received from the engine data bus">
      <div className="telemetry">
        {fields.map(([key, label, unit, digits = 1]) => (
          <span key={key}>
            <small>
              {label} {unit && <i>{unit}</i>}
            </small>
            {telemetry?.[key]?.toFixed?.(digits) ?? telemetry?.[key] ?? '--'}
          </span>
        ))}
      </div>
    </Panel>
  )
}

export function Advisory({ snapshot }) {
  const advisory = snapshot?.advisory
  const rul = snapshot?.rul
  if (!advisory) return null
  const tone =
    advisory.dispatch === 'GO' ? 'ok' : advisory.dispatch === 'NO_GO' ? 'bad' : 'warn'
  return (
    <section className={`advisory ${tone}`}>
      <Gauge size={26} />
      <div>
        <small>MISSION DISPATCH DECISION</small>
        <strong>{advisory.dispatch.replaceAll('_', ' ')}</strong>
        <p>{advisory.summary}</p>
      </div>
      <div className="maintenance">
        <small>
          <Wrench size={12} /> MAINTENANCE ADVISORY - {advisory.maintenance_priority}
        </small>
        <p>{advisory.maintenance_action}</p>
        <em>
          Limiting subsystem: {pretty(advisory.limiting_subsystem)}
          {advisory.due_in_hours != null && ` - due within ${advisory.due_in_hours} h`}
          {rul?.status === 'establishing_trend' && ' - trend still being established'}
        </em>
      </div>
    </section>
  )
}

export function AlarmLog({ alarms }) {
  return (
    <Panel title="Alarm and state-change log" subtitle="every confirmed diagnosis transition">
      <div className="alarms">
        {(alarms ?? []).slice(0, 12).map((entry) => (
          <div key={entry.sequence}>
            <b>{entry.engine_hours?.toFixed?.(1)} h</b>
            <span>
              {pretty(entry.from)} → <strong>{pretty(entry.to)}</strong>
            </span>
            <small>
              health {entry.health} - {entry.triggers?.map((t) => t.replaceAll('_', ' ')).join(', ') || 'cleared'}
            </small>
          </div>
        ))}
        {!alarms?.length && <Empty>No state changes recorded.</Empty>}
      </div>
    </Panel>
  )
}
