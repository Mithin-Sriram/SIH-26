import { useEffect, useState } from 'react'

function FlameIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 text-white" fill="currentColor">
      <path d="M12 2c1.5 3.5-.5 5-1.5 6.5C9.2 10.4 8 12 8 14a4 4 0 0 0 8 0c0-1.2-.4-2.2-1-3 2.5 1 4 3.4 4 6a7 7 0 1 1-14 0c0-4.5 3.5-7 5-9 .9-1.2 1.4-3.5 2-6z" />
    </svg>
  )
}

function MenuIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M3 12h18M3 6h18M3 18h18" />
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4 text-slate-400" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </svg>
  )
}

function RefreshIcon({ spinning }) {
  return (
    <svg viewBox="0 0 24 24" className={`h-4 w-4 ${spinning ? 'animate-spin' : ''}`}
      fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M21 12a9 9 0 1 1-2.64-6.36" />
      <path d="M21 3v6h-6" />
    </svg>
  )
}

function Freshness({ lastRefresh }) {
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  if (!lastRefresh) {
    return (
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span className="h-2 w-2 rounded-full bg-slate-300" />
        <span className="hidden sm:inline">connecting…</span>
      </div>
    )
  }
  const secs = Math.max(0, Math.floor((now - lastRefresh.getTime()) / 1000))
  const fresh = secs < 70
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500">
      <span className={`h-2 w-2 rounded-full ${
        fresh ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
      }`} />
      <span className="font-mono hidden sm:inline">
        {fresh ? 'live' : 'stale'} · {secs}s ago
      </span>
      <span className="font-mono sm:hidden">{fresh ? 'live' : `${secs}s`}</span>
    </div>
  )
}

export default function TopNav({ search, onSearch, lastRefresh, onRefresh,
                                refreshing, sidebarOpen, onToggleSidebar }) {
  const [searchOpen, setSearchOpen] = useState(false)

  return (
    <header className="h-14 shrink-0 bg-white border-b border-slate-200
      flex items-center gap-2 sm:gap-4 px-3 sm:px-4 shadow-sm">
      <button
        onClick={onToggleSidebar}
        className="md:hidden h-9 w-9 rounded-lg flex items-center justify-center
          text-slate-500 hover:bg-slate-100 transition"
        aria-label="Toggle sidebar"
      >
        <MenuIcon />
      </button>

      <div className="flex items-center gap-2 sm:gap-3">
        <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-red-600
          to-orange-500 flex items-center justify-center shadow-sm shrink-0">
          <FlameIcon />
        </div>
        <div className="leading-tight">
          <h1 className="text-[14px] sm:text-[15px] font-semibold text-slate-900">
            Thermal Anomaly Classifier
          </h1>
          <p className="text-[10px] sm:text-[11px] text-slate-400">
            SIH26162 · fire type intelligence
          </p>
        </div>
      </div>

      {/* Desktop search */}
      <div className="relative ml-4 sm:ml-6 flex-1 max-w-md hidden md:block">
        <div className="absolute left-3 top-1/2 -translate-y-1/2">
          <SearchIcon />
        </div>
        <input
          type="text"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          placeholder="Search id, class, notes, coordinates…"
          className="w-full h-9 pl-9 pr-3 rounded-lg bg-slate-100
            border border-slate-200 text-sm text-slate-700
            placeholder:text-slate-400 focus:outline-none
            focus:ring-2 focus:ring-orange-400/50 focus:border-orange-400
            focus:bg-white transition"
        />
      </div>

      {/* Mobile search toggle */}
      <div className="md:hidden ml-auto">
        <button
          onClick={() => setSearchOpen((v) => !v)}
          className="h-9 w-9 rounded-lg flex items-center justify-center
            text-slate-500 hover:bg-slate-100 transition"
          aria-label="Toggle search"
        >
          <SearchIcon />
        </button>
      </div>

      {/* Mobile search bar (expandable) */}
      {searchOpen && (
        <div className="md:hidden absolute top-14 left-0 right-0 z-50
          bg-white border-b border-slate-200 px-3 py-2 shadow-md">
          <div className="relative">
            <div className="absolute left-3 top-1/2 -translate-y-1/2">
              <SearchIcon />
            </div>
            <input
              type="text"
              value={search}
              onChange={(e) => onSearch(e.target.value)}
              placeholder="Search id, class, notes, coordinates…"
              className="w-full h-9 pl-9 pr-3 rounded-lg bg-slate-100
                border border-slate-200 text-sm text-slate-700
                placeholder:text-slate-400 focus:outline-none
                focus:ring-2 focus:ring-orange-400/50 focus:border-orange-400
                focus:bg-white transition"
              autoFocus
            />
          </div>
        </div>
      )}

      <div className="ml-auto flex items-center gap-3 sm:gap-4">
        <Freshness lastRefresh={lastRefresh} />
        <button
          onClick={onRefresh}
          disabled={refreshing}
          className="h-9 px-3 rounded-lg border border-slate-200 text-sm
            text-slate-600 hover:bg-slate-50 hover:border-slate-300
            disabled:opacity-50 flex items-center gap-1.5 transition"
        >
          <RefreshIcon spinning={refreshing} />
          <span className="hidden sm:inline">Refresh</span>
        </button>
      </div>
    </header>
  )
}
