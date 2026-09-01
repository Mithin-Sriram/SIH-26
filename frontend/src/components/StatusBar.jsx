export default function StatusBar({ shown, total, lastRefresh, error }) {
  const time = lastRefresh
    ? lastRefresh.toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      })
    : '—'

  return (
    <footer className="h-8 shrink-0 bg-white border-t border-slate-200
      flex items-center justify-between px-3 sm:px-4 text-xs text-slate-500">
      <div className="flex items-center gap-2">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            error ? 'bg-red-500' : 'bg-emerald-500'
          }`}
        />
        <span>
          <span className="font-mono text-slate-700">{shown}</span>
          <span className="hidden sm:inline"> of </span>
          <span className="sm:hidden">/</span>
          <span className="font-mono text-slate-700">{total}</span>
          <span className="hidden sm:inline"> detections</span>
        </span>
      </div>
      <div className="flex items-center gap-3">
        <span className="hidden sm:inline">auto-refresh 60s</span>
        <span>
          <span className="hidden sm:inline">Last refresh </span>
          <span className="font-mono text-slate-700">{time}</span>
        </span>
      </div>
    </footer>
  )
}
