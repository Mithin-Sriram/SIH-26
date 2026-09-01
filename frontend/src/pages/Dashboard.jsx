import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import TopNav from '../components/TopNav'
import Sidebar from '../components/Sidebar'
import MapView from '../components/MapView'
import DetailPanel from '../components/DetailPanel'
import StatusBar from '../components/StatusBar'
import { CLASSES, REFRESH_INTERVAL_MS } from '../lib/constants'
import * as api from '../lib/api'

export default function Dashboard() {
  const [detections, setDetections] = useState(null)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)
  const [lastRefresh, setLastRefresh] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [filters, setFilters] = useState(() => new Set(CLASSES))
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const detailRef = useRef({})
  const [, setDetailTick] = useState(0)
  const bumpDetail = () => setDetailTick((n) => n + 1)

  const refresh = useCallback(async () => {
    setRefreshing(true)
    try {
      const [d, s] = await Promise.all([
        api.getDetections({ limit: 2000 }),
        api.getStats(),
      ])
      setDetections(d)
      setStats(s)
      setError(null)
      setLastRefresh(new Date())
    } catch (e) {
      setError(`Is the FastAPI server running on localhost:8000? (${e.message})`)
    } finally {
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_INTERVAL_MS)
    return () => clearInterval(id)
  }, [refresh])

  useEffect(() => {
    if (!selectedId || detailRef.current[selectedId]) return
    detailRef.current[selectedId] = { status: 'loading' }
    bumpDetail()
    api.getDetection(selectedId)
      .then((data) => {
        detailRef.current[selectedId] = { status: 'ok', data }
        bumpDetail()
      })
      .catch((e) => {
        detailRef.current[selectedId] = { status: 'error', message: e.message }
        bumpDetail()
      })
  }, [selectedId])

  const filtered = useMemo(() => {
    if (!detections) return []
    const q = search.trim().toLowerCase()
    return detections.filter((d) => {
      if (!filters.has(d.category)) return false
      if (!q) return true
      return [d.id, d.category, d.notes, d.source,
        d.latitude.toFixed(4), d.longitude.toFixed(4)]
        .some((s) => String(s).toLowerCase().includes(q))
    })
  }, [detections, filters, search])

  const toggleClass = (c) => {
    setFilters((prev) => {
      const next = new Set(prev)
      if (next.has(c)) next.delete(c)
      else next.add(c)
      return next
    })
  }

  const clearFilters = () => {
    setFilters(new Set(CLASSES))
    setSearch('')
  }

  const empty = detections !== null && filtered.length === 0

  return (
    <div className="h-full flex flex-col bg-slate-100 font-sans">
      <TopNav
        search={search}
        onSearch={setSearch}
        lastRefresh={lastRefresh}
        onRefresh={refresh}
        refreshing={refreshing}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
      />

      <div className="flex flex-1 min-h-0 relative">
        <Sidebar
          stats={stats}
          filters={filters}
          onToggleClass={toggleClass}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        <main className="relative flex-1 min-w-0 z-0">
          <MapView
            detections={filtered}
            selectedId={selectedId}
            onSelect={setSelectedId}
            loading={detections === null}
            error={error}
            empty={empty}
            onClearFilters={clearFilters}
          />
        </main>
      </div>

      {selectedId && (
        <DetailPanel
          entry={detailRef.current[selectedId]}
          onClose={() => setSelectedId(null)}
        />
      )}

      <StatusBar
        shown={filtered.length}
        total={detections?.length ?? 0}
        lastRefresh={lastRefresh}
        error={error}
      />
    </div>
  )
}
