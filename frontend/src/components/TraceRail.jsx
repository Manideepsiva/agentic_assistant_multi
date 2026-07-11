import { useState } from 'react'

export default function TraceRail({ trace, extracted, cost, open }) {
  const [tab, setTab] = useState('trace')

  return (
    <aside className={`rail ${open ? 'open' : ''}`}>
      <div className="rail-tabs">
        <button className={tab === 'trace' ? 'active' : ''} onClick={() => setTab('trace')}>
          Plan trace
        </button>
        <button className={tab === 'extract' ? 'active' : ''} onClick={() => setTab('extract')}>
          Extracted text
        </button>
      </div>

      {cost && (
        <div className="cost-chip">
          est. cost <b>${cost.usd}</b> · ~{cost.input_tokens.toLocaleString()} in / {cost.output_tokens.toLocaleString()} out tokens
        </div>
      )}

      {tab === 'trace' && (
        <div className="rail-pane">
          {trace.length === 0 ? (
            <div className="rail-empty">The agent's plan and tool calls will appear here in real time.</div>
          ) : (
            trace.map((ev, i) => (
              <div key={i} className={`trace-item ${ev.status}`}>
                <div className="stage">{ev.stage}</div>
                <div className="title">{ev.title}</div>
                {ev.detail && <div className="detail">{ev.detail}</div>}
              </div>
            ))
          )}
        </div>
      )}

      {tab === 'extract' && (
        <div className="rail-pane">
          {extracted.length === 0 ? (
            <div className="rail-empty">Extracted OCR / PDF / audio text will appear here.</div>
          ) : (
            extracted.map((it, i) => {
              let meta = `${it.modality} · ${it.method}`
              if (it.confidence != null) meta += ` · OCR conf ${it.confidence}%`
              if (it.meta?.duration_seconds) meta += ` · ${it.meta.duration_seconds}s`
              if (it.meta?.pages) meta += ` · ${it.meta.pages} page(s)`
              return (
                <div className="source-block" key={i}>
                  <h4>{it.source}</h4>
                  <div className="meta">{meta}</div>
                  <pre>{it.content || '(no text extracted)'}</pre>
                </div>
              )
            })
          )}
        </div>
      )}
    </aside>
  )
}
