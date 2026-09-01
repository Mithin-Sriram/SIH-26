import { useEffect } from 'react'
import { MapContainer, TileLayer, CircleMarker, Tooltip, useMap }
  from 'react-leaflet'
import {
  CATEGORY_COLORS,
  INDIA_CENTER,
  INDIA_ZOOM,
  SHORT_NAMES,
} from '../lib/constants'

function AutoResize() {
  const map = useMap()
  useEffect(() => {
    const container = map.getContainer()
    const observer = new ResizeObserver(() => map.invalidateSize())
    observer.observe(container)
    return () => observer.disconnect()
  }, [map])
  return null
}

function radiusFor(frp, selected) {
  return (selected ? 3 : 0) + 4 + Math.min(10, Math.sqrt(frp || 0) / 7)
}

function Overlay({ children }) {
  return (
    <div className="absolute inset-0 z-[1100] flex items-center
      justify-center bg-slate-100/70 backdrop-blur-[1px] pointer-events-none">
      <div className="pointer-events-auto bg-white rounded-xl shadow-lg
        border border-slate-200 px-6 py-5 max-w-xs text-center">
        {children}
      </div>
    </div>
  )
}

export default function MapView({ detections, selectedId, onSelect,
                                 loading, error, empty, onClearFilters }) {
  return (
    <div className="absolute inset-0">
      <MapContainer
        center={INDIA_CENTER}
        zoom={INDIA_ZOOM}
        minZoom={4}
        scrollWheelZoom
        className="h-full w-full"
      >
        <AutoResize />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {(detections ?? []).map((d) => {
          const selected = d.id === selectedId
          const color = CATEGORY_COLORS[d.category] || '#6b7280'
          return (
            <CircleMarker
              key={d.id}
              center={[d.latitude, d.longitude]}
              radius={radiusFor(d.frp_mw, selected)}
              pathOptions={{
                color: selected ? '#0f172a' : '#334155',
                weight: selected ? 3 : 1,
                fillColor: color,
                fillOpacity: selected ? 0.95 : 0.72,
                opacity: 1,
              }}
              eventHandlers={{ click: () => onSelect(d.id) }}
            >
              <Tooltip direction="top" offset={[0, -6]}>
                <span className="font-semibold text-slate-900">
                  {SHORT_NAMES[d.category] || d.category}
                </span>
                <br />
                <span className="font-mono">
                  {Math.round(d.category_probability * 100)}% · {d.frp_mw} MW
                </span>
                <br />
                <span className="font-mono text-slate-500">{d.id}</span>
              </Tooltip>
            </CircleMarker>
          )
        })}
      </MapContainer>

      {loading && (
        <Overlay>
          <div className="flex flex-col items-center gap-3">
            <div className="h-7 w-7 rounded-full border-[3px]
              border-slate-200 border-t-orange-500 animate-spin" />
            <p className="text-sm text-slate-500">
              Loading detections from the classifier…
            </p>
          </div>
        </Overlay>
      )}

      {!loading && error && (
        <Overlay>
          <p className="text-sm font-semibold text-red-600">
            Backend unreachable
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {error}
          </p>
        </Overlay>
      )}

      {!loading && !error && empty && (
        <Overlay>
          <p className="text-sm font-semibold text-slate-700">
            No detections to show
          </p>
          <p className="text-xs text-slate-500 mt-1">
            All classes filtered out — enable a class to see markers again.
          </p>
          <button
            onClick={onClearFilters}
            className="mt-3 text-xs font-medium text-orange-600
              hover:text-orange-700 underline underline-offset-2"
          >
            Clear filters
          </button>
        </Overlay>
      )}
    </div>
  )
}
