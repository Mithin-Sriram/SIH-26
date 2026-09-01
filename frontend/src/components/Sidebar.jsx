import { CLASSES, SHORT_NAMES, CATEGORY_COLORS } from '../lib/constants'

export default function Sidebar({ stats, filters, onToggleClass, open, onClose }) {
  const byClass = stats?.by_class ?? {}
  const sources = stats?.sources ?? {}

  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-black/40 backdrop-blur-[1px]"
          onClick={onClose}
        />
      )}

      <aside className={`
        shrink-0 bg-slate-900 text-slate-200 flex flex-col
        fixed md:static top-14 bottom-0 left-0 z-50
        w-[260px] md:w-[230px] lg:w-[260px] xl:w-[300px]
        transition-transform duration-200 ease-in-out
        ${open ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        {/* Mobile close button */}
        <div className="md:hidden px-4 pt-3 pb-1 flex justify-end">
          <button
            onClick={onClose}
            className="h-8 w-8 rounded-lg flex items-center justify-center
              text-slate-400 hover:bg-slate-800 transition"
            aria-label="Close sidebar"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-4 pt-2 md:pt-4 pb-2">
          <h2 className="text-[11px] font-semibold tracking-widest
            uppercase text-slate-500">Filter by class</h2>
        </div>

        <div className="px-2 space-y-0.5">
          {CLASSES.map((c) => {
            const on = filters.has(c)
            return (
              <label
                key={c}
                className={`flex items-center gap-2.5 px-2.5 py-2 rounded-lg
                  cursor-pointer select-none transition
                  ${on ? 'hover:bg-slate-800' : 'hover:bg-slate-800 opacity-50'}`}
              >
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() => onToggleClass(c)}
                  className="h-4 w-4 rounded cursor-pointer"
                  style={{ accentColor: CATEGORY_COLORS[c] }}
                />
                <span
                  className="h-2.5 w-2.5 rounded-full shrink-0"
                  style={{ backgroundColor: CATEGORY_COLORS[c] }}
                />
                <span className="text-[13px] flex-1 text-slate-200">
                  {SHORT_NAMES[c]}
                </span>
                <span className="font-mono text-xs text-slate-400">
                  {byClass[c] ?? '–'}
                </span>
              </label>
            )
          })}
        </div>

        <div className="mt-auto border-t border-slate-800 px-4 py-3 space-y-2
          text-xs text-slate-400">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-red-500" />
              High priority
            </span>
            <span className="font-mono text-slate-200">
              {stats?.high_priority ?? '–'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Total detections</span>
            <span className="font-mono text-slate-200">
              {stats?.total ?? '–'}
            </span>
          </div>
          {Object.keys(sources).length > 0 && (
            <div className="flex items-center justify-between">
              <span>Sources</span>
              <span className="font-mono text-slate-300">
                {Object.entries(sources)
                  .map(([k, v]) => `${k} ${v}`)
                  .join(' · ')}
              </span>
            </div>
          )}
        </div>
      </aside>
    </>
  )
}
