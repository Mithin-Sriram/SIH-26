import { API_BASE } from './constants'

async function json(res) {
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    const msg = detail?.detail?.message || detail?.detail || `HTTP ${res.status}`
    throw new Error(typeof msg === 'string' ? msg : `HTTP ${res.status}`)
  }
  return res.json()
}

export function getDetections(params = {}) {
  const qs = new URLSearchParams(params).toString()
  return fetch(`${API_BASE}/detections${qs ? `?${qs}` : ''}`).then(json)
}

export function getStats() {
  return fetch(`${API_BASE}/stats`).then(json)
}

export function getDetection(id) {
  return fetch(`${API_BASE}/detections/${id}`).then(json)
}
