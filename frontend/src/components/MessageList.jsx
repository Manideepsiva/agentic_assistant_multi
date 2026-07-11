// Renders **bold** as <b>, everything else stays plain text (output is text-only).
function renderText(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <b key={i}>{part.slice(2, -2)}</b>
    }
    return part
  })
}

export default function MessageList({ messages, emptyState }) {
  if (messages.length === 0) {
    return (
      <div className="messages">
        <div className="empty">
          <h2>Drop in anything. Ask for anything.</h2>
          <p>{emptyState}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="messages" id="messages">
      {messages.map((m, i) => (
        <div key={i} className={`msg ${m.role} ${m.kind === 'clarify' ? 'clarify' : ''} ${m.kind === 'error' ? 'error' : ''}`}>
          <div className="role">{m.role === 'user' ? 'You' : 'Agent'}</div>
          <div>{renderText(m.text)}</div>
        </div>
      ))}
    </div>
  )
}
