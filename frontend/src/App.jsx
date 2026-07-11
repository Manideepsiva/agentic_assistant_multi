import { useState } from 'react'
import Composer from './components/Composer.jsx'
import MessageList from './components/MessageList.jsx'
import TraceRail from './components/TraceRail.jsx'
import { streamChat } from './api.js'

export default function App() {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [trace, setTrace] = useState([])
  const [extracted, setExtracted] = useState([])
  const [cost, setCost] = useState(null)
  const [busy, setBusy] = useState(false)
  const [railOpen, setRailOpen] = useState(false)

  async function handleSend(query, files) {
    setBusy(true)
    setTrace([])
    setCost(null)

    const userLabel = query + (files.length ? `\n📎 ${files.map((f) => f.name).join(', ')}` : '')
    setMessages((prev) => [...prev, { role: 'user', text: userLabel }])
    setMessages((prev) => [...prev, { role: 'agent', text: '', kind: 'answer' }])

    let answerText = ''

    try {
      await streamChat({ query, sessionId, files }, (ev) => {
        if (ev.type === 'session') {
          setSessionId(ev.session_id)
        } else if (ev.type === 'trace') {
          setTrace((prev) => [...prev, ev.event])
        } else if (ev.type === 'extracted') {
          setExtracted(ev.items)
        } else if (ev.type === 'cost') {
          setCost(ev.estimate)
        } else if (ev.type === 'token') {
          answerText += ev.text
          setMessages((prev) => {
            const next = [...prev]
            next[next.length - 1] = { ...next[next.length - 1], text: answerText }
            return next
          })
        } else if (ev.type === 'done') {
          const finalText = ev.answer || answerText || '(no answer)'
          setMessages((prev) => {
            const next = [...prev]
            next[next.length - 1] = { role: 'agent', text: finalText, kind: ev.kind }
            return next
          })
        }
      })
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev]
        next[next.length - 1] = {
          role: 'agent',
          kind: 'error',
          text: `Request failed: ${err.message}. Check the server logs and your GROQ_API_KEY, then try again.`,
        }
        return next
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <span className="dot" />
        <h1>Agentic Multimodal Assistant</h1>
        <button className="rail-toggle" onClick={() => setRailOpen((v) => !v)}>
          TRACE
        </button>
        <span className="badge">groq · langgraph</span>
      </header>

      <div className="main">
        <section className="chat-col">
          <MessageList
            messages={messages}
            emptyState="Text, images, PDFs and audio — together in one request. The agent extracts everything, plans the minimal tool chain, and executes it autonomously. Watch every step in the trace rail →"
          />
          <Composer onSend={handleSend} busy={busy} />
        </section>

        <TraceRail trace={trace} extracted={extracted} cost={cost} open={railOpen} />
      </div>
    </div>
  )
}
