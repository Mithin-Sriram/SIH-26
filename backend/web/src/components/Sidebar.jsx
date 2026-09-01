import { CLASSES, SHORT_NAMES, CATEGORY_COLORS } from '../lib/constants'

export default function Sidebar({ stats, filters, onToggleClass }) {
  const byClass = stats?.by_class ?? {}
  const sources = stats?.sources ?? {}

  return (
    <aside className="w-1/5 min-w-[230px] max-w-[300px] shrink-0
      bg-slate-900 text-slate-200 flex flex-col">
      <div className="px-4 pt-4 pb-2">
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
  )
}
