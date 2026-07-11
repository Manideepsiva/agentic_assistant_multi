// Talks to /api/chat/stream (Server-Sent Events). Calls onEvent(event) for
// every parsed event: {type:'session'|'trace'|'extracted'|'cost'|'token'|'done', ...}
export async function streamChat({ query, sessionId, files }, onEvent) {
  const fd = new FormData()
  fd.append('query', query || '')
  if (sessionId) fd.append('session_id', sessionId)
  files.forEach((f) => fd.append('files', f))

  const resp = await fetch('/api/chat/stream', { method: 'POST', body: fd })
  if (!resp.ok) {
    throw new Error(await resp.text())
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop()
    for (const raw of chunks) {
      const line = raw.split('\n').find((l) => l.startsWith('data: '))
      if (!line) continue
      onEvent(JSON.parse(line.slice(6)))
    }
  }
}
