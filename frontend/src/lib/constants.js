export const API_BASE = 'http://localhost:8000/api'

export const REFRESH_INTERVAL_MS = 60000

export const CLASSES = [
  'Industrial Fire',
  'Gas Flare',
  'Wildfire',
  'Agricultural Burning',
  'Other/Unknown',
]

export const SHORT_NAMES = {
  'Industrial Fire': 'Industrial',
  'Gas Flare': 'Gas Flare',
  Wildfire: 'Wildfire',
  'Agricultural Burning': 'Ag Burning',
  'Other/Unknown': 'Other',
}

export const CATEGORY_COLORS = {
  'Industrial Fire': '#dc2626',
  'Gas Flare': '#ea580c',
  Wildfire: '#16a34a',
  'Agricultural Burning': '#eab308',
  'Other/Unknown': '#6b7280',
}

export const PRIORITY_STYLES = {
  High: 'bg-red-600 text-white',
  Medium: 'bg-yellow-400 text-slate-900',
  Low: 'bg-slate-500 text-white',
}

export const INDIA_CENTER = [22.6, 81.0]
export const INDIA_ZOOM = 5
