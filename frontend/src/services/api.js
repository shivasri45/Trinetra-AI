// Single place that knows how to reach the backend.
// The API key is supplied through Vite env (VITE_TRINETRA_API_KEY) and falls back
// to the documented development key so a fresh clone runs without setup.

const BASE = import.meta.env.VITE_TRINETRA_API ?? 'http://localhost:8000'
const API_KEY = import.meta.env.VITE_TRINETRA_API_KEY ?? 'trinetra-dev-key'

const authHeaders = () => ({ 'X-API-Key': API_KEY })

async function request(path, options = {}) {
  const response = await fetch(BASE + path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...authHeaders(), ...(options.headers ?? {}) },
  })
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(`${options.method ?? 'GET'} ${path} failed: ${response.status} ${detail}`)
  }
  if (response.status === 204) return null
  return response.json()
}

export const api = {
  base: BASE,
  get: (path) => request(path),
  post: (path, body) => request(path, { method: 'POST', body: body ? JSON.stringify(body) : '{}' }),
  del: (path) => request(path, { method: 'DELETE' }),

  // controls
  start: () => api.post('/simulation/start'),
  stop: () => api.post('/simulation/stop'),
  reset: () => api.post('/simulation/reset'),
  injectFault: (fault) => api.post('/simulation/fault', { fault }),
  setMission: (mission) => api.post('/mission/configure', { mission }),
  setIngestion: (mode) => api.post('/ingestion/mode', { mode }),

  // reads
  capabilities: () => api.get('/system/capabilities'),
  health: () => api.get('/health'),
  snapshot: () => api.get('/telemetry/snapshot'),
  spectrum: () => api.get('/vibration/spectrum'),
  frames: () => api.get('/ingestion/frames'),
  alarms: () => api.get('/alarms'),
  modelMetrics: () => api.get('/model/metrics'),
  interfaceControl: () => api.get('/interface-control'),
  securityPosture: () => api.get('/security/posture'),
  verify: (telemetry, signature) => api.post('/security/verify', { telemetry, signature }),

  // missions
  missions: () => api.get('/missions'),
  missionTimeline: (id, step = 1) => api.get(`/missions/${id}/timeline?step=${step}`),
  missionSnapshots: (id, start = 0, limit = 400) =>
    api.get(`/missions/${id}/snapshots?start=${start}&limit=${limit}`),
  missionReport: (id) => api.get(`/missions/${id}/report`),
  deleteMission: (id) => api.del(`/missions/${id}`),
}

export function telemetrySocket(onSnapshot, onStatus) {
  const url = BASE.replace(/^http/, 'ws') + `/ws/telemetry?api_key=${encodeURIComponent(API_KEY)}`
  let socket
  let closed = false
  let retry

  const connect = () => {
    socket = new WebSocket(url)
    socket.onopen = () => onStatus?.('live')
    socket.onmessage = (event) => onSnapshot(JSON.parse(event.data))
    socket.onerror = () => onStatus?.('error')
    socket.onclose = () => {
      onStatus?.('offline')
      if (!closed) retry = setTimeout(connect, 2000)
    }
  }
  connect()

  return () => {
    closed = true
    clearTimeout(retry)
    socket?.close()
  }
}
