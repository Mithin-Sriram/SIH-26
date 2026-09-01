import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Cell, Tooltip,
} from 'recharts'
import {
  CATEGORY_COLORS, PRIORITY_STYLES, SHORT_NAMES,
} from '../lib/constants'

function Section({ title, children, className = '' }) {
  return (
    <div className={`px-4 py-3.5 border-b border-slate-100 ${className}`}>
      <h4 className="text-[10px] font-semibold tracking-widest uppercase
        text-slate-400 mb-2.5">{title}</h4>
      {children}
    </div>
  )
}

function Reading({ label, value, unit }) {
  return (
    <div className="bg-slate-50 rounded-lg px-3 py-2 border
      border-slate-100">
      <p className="text-[10px] text-slate-400 uppercase tracking-wide">
        {label}
      </p>
      <p className="font-mono text-sm text-slate-800 mt-0.5">
        {value}
        {unit && <span className="text-slate-400 text-xs ml-1">{unit}</span>}
      </p>
    </div>
  )
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  )
}

function PriorityBadge({ priority }) {
  return (
    <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold
      tracking-widest ${PRIORITY_STYLES[priority] || PRIORITY_STYLES.Low}`}>
      {String(priority).toUpperCase()} PRIORITY
    </span>
  )
}

function ProbChart({ top3 }) {
  const data = top3.map((t) => ({
    name: SHORT_NAMES[t.class] || t.class,
    prob: t.probability,
    color: CATEGORY_COLORS[t.class] || '#6b7280',
  }))
  return (
    <ResponsiveContainer width="100%" height={104}>
      <BarChart data={data} layout="vertical"
        margin={{ top: 0, right: 24, bottom: 0, left: 0 }} barSize={16}>
        <XAxis type="number" domain={[0, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fontSize: 10, fill: '#94a3b8' }}
          axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="name" width={82}
          tick={{ fontSize: 11, fill: '#334155' }}
          axisLine={false} tickLine={false} />
        <Tooltip
          formatter={(v) => [`${(v * 100).toFixed(1)}%`, 'probability']}
          contentStyle={{
            borderRadius: 8, border: '1px solid #e2e8f0',
            fontSize: 11, boxShadow: '0 4px 12px rgb(15 23 42 / 0.08)',
          }}
          cursor={{ fill: 'rgb(15 23 42 / 0.04)' }}
        />
        <Bar dataKey="prob" radius={[0, 6, 6, 0]} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function DetailBody({ d }) {
  const color = CATEGORY_COLORS[d.predicted_class] || '#6b7280'
  const whyNotBullets = d.why_not
    ? d.why_not.explanation.split('; ').filter((s) => s.trim())
    : []
  const i5 = d.features?.brightness_temp_i5_k

  return (
    <>
      <div className="px-4 pt-4 pb-3.5 border-b border-slate-100">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2.5">
            <span className="h-3 w-3 rounded-full shrink-0"
              style={{ backgroundColor: color }} />
            <h3 className="font-semibold text-slate-900 leading-tight">
              {d.predicted_class}
            </h3>
          </div>
          <span className="font-mono text-[10px] text-slate-400 mt-0.5">
            {d.id}
          </span>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <PriorityBadge priority={d.priority} />
          <span className="font-mono text-xs text-slate-500">
            {Math.round(d.probability * 100)}% confident
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-2 leading-relaxed">
          {d.priority_reason}
        </p>
      </div>

      <Section title="Location">
        <div className="space-y-1.5 text-xs text-slate-600">
          <p className="font-mono text-sm text-slate-800">
            {d.latitude.toFixed(4)}°N, {d.longitude.toFixed(4)}°E
          </p>
          <p>Detected {new Date(d.detected_at).toLocaleString()}</p>
          <p className="flex items-center gap-2">
            <span className="px-1.5 py-0.5 rounded bg-slate-100 border
              border-slate-200 text-[10px] font-medium text-slate-500
              uppercase tracking-wide">{d.source}</span>
          </p>
          {d.nearest_industrial && (
            <p className="flex items-center gap-1.5 pt-1">
              <span className="text-slate-400">Nearest facility:</span>
              <span className="text-slate-700">{d.nearest_industrial.name}</span>
              <span className="font-mono text-slate-500">
                ({(d.nearest_industrial.distance_m / 1000).toFixed(0)} km)
              </span>
            </p>
          )}
        </div>
      </Section>

      <Section title="Thermal readings">
        <div className="grid grid-cols-2 gap-2">
          <Reading label="FRP" value={d.frp_mw.toFixed(1)} unit="MW" />
          <Reading label="Brightness I4" value={
            Math.round(d.brightness_temp_k)} unit="K" />
          {i5 != null && (
            <Reading label="Brightness I5" value={
              Math.round(i5)} unit="K" />
          )}
          <Reading label="Confidence" value={
            Math.round(d.confidence * 100)} unit="%" />
        </div>
      </Section>

      <Section title="Classification (calibrated)">
        <ProbChart top3={d.top_3} />
      </Section>

      <Section title="Key evidence">
        <ul className="space-y-2">
          {d.evidence.map((e, i) => (
            <li key={i} className="flex gap-2 text-xs text-slate-600
              leading-relaxed">
              <span className="mt-1 h-1.5 w-1.5 rounded-full shrink-0"
                style={{ backgroundColor: color }} />
              <span>{e}</span>
            </li>
          ))}
        </ul>
      </Section>

      {d.why_not && (
        <Section title={`Why not ${SHORT_NAMES[d.why_not.class_name]
          || d.why_not.class_name}?`}>
          <ul className="space-y-2">
            {whyNotBullets.map((b, i) => (
              <li key={i} className="flex gap-2 text-xs text-slate-600
                leading-relaxed">
                <svg viewBox="0 0 24 24" className="h-3 w-3 mt-0.5 shrink-0
                  text-slate-400" fill="none" stroke="currentColor"
                  strokeWidth="2.5" strokeLinecap="round">
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
                <span>{b.replace(/\.$/, '')}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      <div className="px-4 py-3 text-[10px] font-mono text-slate-400">
        {d.model_version} · {d.features
          ? Object.keys(d.features).length + ' features'
          : ''}
      </div>
    </>
  )
}

export default function DetailPanel({ entry, onClose }) {
  return (
    <aside className="w-1/5 min-w-[270px] max-w-[360px] shrink-0 bg-white
      border-l border-slate-200 overflow-y-auto relative">
      <button
        onClick={onClose}
        className="absolute top-3 right-3 z-10 h-7 w-7 rounded-lg
          flex items-center justify-center text-slate-400
          hover:bg-slate-100 hover:text-slate-600 transition"
        aria-label="Close detail panel"
      >
        <CloseIcon />
      </button>

      {(!entry || entry.status === 'loading') && (
        <div className="h-full flex flex-col items-center justify-center
          gap-3 text-slate-400">
          <div className="h-7 w-7 rounded-full border-[3px]
            border-slate-200 border-t-orange-500 animate-spin" />
          <p className="text-xs">Classifying detection…</p>
        </div>
      )}

      {entry?.status === 'error' && (
        <div className="h-full flex flex-col items-center justify-center
          gap-2 px-6 text-center">
          <p className="text-sm font-semibold text-red-600">
            Could not load detail
          </p>
          <p className="text-xs text-slate-500">{entry.message}</p>
        </div>
      )}

      {entry?.status === 'ok' && <DetailBody d={entry.data} />}
    </aside>
  )
}
