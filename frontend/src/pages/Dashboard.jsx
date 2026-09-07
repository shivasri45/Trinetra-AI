import React from 'react'
import { CylinderChart, RulChart, SpectrumChart, TrendChart } from '../components/charts.jsx'
import {
  Advisory, AlarmLog, Diagnosis, HealthSummary, ResidualTable, SubsystemHealth, TelemetryGrid,
} from '../components/panels.jsx'

const RESIDUAL_CHANNELS = [
  'rpm', 'cht', 'egt', 'oil_pressure', 'oil_temperature', 'fuel_flow',
  'air_fuel_ratio', 'vibration', 'alternator_output',
]

const TELEMETRY_FIELDS = [
  ['rpm', 'SPEED', 'rpm', 0],
  ['cht', 'CHT', '\u00b0C', 1],
  ['egt', 'EGT', '\u00b0C', 1],
  ['egt_spread', 'EGT SPREAD', '\u00b0C', 1],
  ['oil_pressure', 'OIL PRESS', 'psi', 1],
  ['oil_temperature', 'OIL TEMP', '\u00b0C', 1],
  ['fuel_flow', 'FUEL FLOW', 'L/h', 2],
  ['air_fuel_ratio', 'AFR', ':1', 2],
  ['manifold_pressure', 'MAP', 'kPa', 1],
  ['injection_timing', 'INJ ADVANCE', '\u00b0BTDC', 1],
  ['vibration', 'VIBRATION', 'g', 3],
  ['battery_voltage', 'BATTERY', 'V', 2],
  ['alternator_output', 'ALTERNATOR', 'A', 1],
  ['power_kw', 'SHAFT POWER', 'kW', 1],
  ['pressure_altitude', 'PRESS ALT', 'm', 0],
  ['ambient_temperature', 'OAT', '\u00b0C', 1],
]

export default function Dashboard({ snapshot, history, spectrum, alarms }) {
  const twin = snapshot?.digital_twin
  return (
    <>
      <HealthSummary snapshot={snapshot} />
      <Advisory snapshot={snapshot} />

      <div className="grid">
        <div className="column">
          <TrendChart
            title="Rotational and thermal"
            subtitle="RPM, cylinder head and exhaust gas temperature"
            history={history}
            series={[
              ['rpm', '#40c9ff', 'RPM'],
              ['cht', '#ffb25b', 'CHT °C'],
              ['egt', '#fb6376', 'EGT °C'],
            ]}
          />
          <TrendChart
            title="Lubrication and fuel"
            subtitle="oil pressure and temperature, fuel flow"
            history={history}
            series={[
              ['oil_pressure', '#6de6a9', 'Oil psi'],
              ['oil_temperature', '#f4c95d', 'Oil °C'],
              ['fuel_flow', '#ad8dff', 'Fuel L/h'],
            ]}
          />
          <div className="pair">
            <CylinderChart
              measured={snapshot?.cylinders}
              expected={twin?.cylinders?.expected}
              hottest={snapshot?.telemetry?.hottest_cylinder}
              coldest={snapshot?.telemetry?.coldest_cylinder}
            />
            <SpectrumChart spectrum={spectrum} indicators={snapshot?.vibration} />
          </div>
          <div className="pair">
            <TrendChart
              title="Health and anomaly trend"
              subtitle="fused health index against anomaly score"
              history={history}
              height={170}
              series={[
                ['health', '#6de6a9', 'Health'],
                ['anomaly_score', '#fb6376', 'Anomaly'],
              ]}
            />
            <RulChart rul={snapshot?.rul} />
          </div>
        </div>

        <aside className="column">
          <Diagnosis snapshot={snapshot} />
          <SubsystemHealth twin={twin} />
          <ResidualTable twin={twin} channels={RESIDUAL_CHANNELS} />
          <AlarmLog alarms={alarms} />
        </aside>
      </div>

      <TelemetryGrid telemetry={snapshot?.telemetry} fields={TELEMETRY_FIELDS} />
    </>
  )
}
