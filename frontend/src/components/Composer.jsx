import { useRef, useState } from 'react'

export default function Composer({ onSend, busy }) {
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState([])
  const fileInputRef = useRef(null)

  function handleFileChange(e) {
    setFiles((prev) => [...prev, ...Array.from(e.target.files)])
    e.target.value = ''
  }

  function removeFile(i) {
    setFiles((prev) => prev.filter((_, idx) => idx !== i))
  }

  function handleSend() {
    if (busy) return
    if (!query.trim() && files.length === 0) return
    onSend(query.trim(), files)
    setQuery('')
    setFiles([])
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="composer">
      {files.length > 0 && (
        <div className="file-chips">
          {files.map((f, i) => (
            <span className="chip" key={i}>
              {f.name}
              <button onClick={() => removeFile(i)}>✕</button>
            </span>
          ))}
        </div>
      )}
      <div className="input-row">
        <button
          className="btn attach-btn"
          onClick={() => fileInputRef.current?.click()}
          title="Attach files"
        >
          📎 Files
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*,.pdf,audio/*,.mp3,.wav,.m4a"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        <textarea
          className="query-box"
          rows={1}
          placeholder="Ask anything about your files… or just say hi"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button className="btn send-btn" onClick={handleSend} disabled={busy}>
          Send
        </button>
      </div>
      <div className="hint">
        Supports JPG · PNG · PDF · MP3 · WAV · M4A — multiple files at once. Enter to send, Shift+Enter for newline.
      </div>
    </div>
  )
}
