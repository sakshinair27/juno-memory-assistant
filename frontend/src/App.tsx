import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ChatResponse, type Memory, type MemoryEvent, type Quarantined, type Role, type Task } from "./api";
import { speak, startRecording, stopSpeaking, sttSupported, type Recording } from "./voice";

interface Msg {
  role: Role;
  content: string;
  meta?: Omit<ChatResponse, "reply">;
  error?: boolean;
  at?: string; // local date sent, YYYY-MM-DD
}

// Local calendar date; sent with history so "Friday" in an old message is read
// relative to the day it was said, not today.
const today = () => new Date().toLocaleDateString("en-CA");

const newSessionId = () => `s-${Date.now().toString(36)}`;

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable */
  }
}

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => load("sessionId", newSessionId()));
  // Restore the conversation after a refresh, minus any error bubbles (they're only meaningful once).
  const [messages, setMessages] = useState<Msg[]>(() =>
    load<Msg[]>(`msgs:${load("sessionId", "")}`, []).filter((m) => !m.error)
      // a write still "pending" when the page closed can't be watched any more
      .map((m) => (m.meta?.memory_pending ? { ...m, meta: { ...m.meta, memory_pending: false } } : m)));
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [events, setEvents] = useState<MemoryEvent[]>([]);
  const [blocked, setBlocked] = useState<Quarantined[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [voiceOut, setVoiceOut] = useState<boolean>(() => load("voiceOut", false));
  const [voice, setVoice] = useState({ whisper: false, elevenlabs: false });
  const [recording, setRecording] = useState<Recording | null>(null);
  const [flash, setFlash] = useState<Set<string>>(new Set());
  const [panelOpen, setPanelOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [m, e, t, q] = await Promise.all([api.memories(), api.events(), api.tasks(), api.quarantine()]);
      setBlocked(q);
      setMemories(m);
      setEvents(e);
      setTasks(t);
    } catch {
      /* backend not up yet */
    }
  }, []);

  useEffect(() => {
    refresh();
    api.voiceStatus().then(setVoice).catch(() => {});
  }, [refresh]);
  // Effects use block bodies: some browsers return a Promise from scrollIntoView,
  // and React treats any non-function return value as a (broken) cleanup.
  useEffect(() => { save("sessionId", sessionId); }, [sessionId]);
  useEffect(() => { save(`msgs:${sessionId}`, messages.filter((m) => !m.error)); }, [messages, sessionId]);
  useEffect(() => { save("voiceOut", voiceOut); }, [voiceOut]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || busy) return;
    stopSpeaking();
    // Send each past reply's real tool calls along with its text, so the model can tell
    // "I added the reminder" (and did) from a claim it can't back up.
    const history = messages.filter((m) => !m.error).map(({ role, content, meta, at }) => ({
      role, content, at,
      tool_calls: (meta?.tool_calls ?? []).map(({ name, input, output }) => ({ name, input, output })),
    }));
    setMessages((m) => [...m, { role: "user", content, at: today() }]);
    setInput("");
    setBusy(true);
    try {
      const { reply, ...meta } = await api.chat(sessionId, content, history);
      setMessages((m) => [...m, { role: "assistant", content: reply, meta, at: today() }]);
      if (voiceOut) speak(reply, voice.elevenlabs);
      refresh();
      if (meta.memory_pending && meta.turn_id) watchMemory(meta.turn_id);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: `Something went wrong: ${(e as Error).message}`, error: true }]);
    } finally {
      setBusy(false);
    }
  }

  // The reply arrives before memory is written; poll until the background write
  // lands, then attach its result to that reply and refresh the panel.
  async function watchMemory(turnId: string) {
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 750));
      let res;
      try {
        res = await api.turnMemory(turnId);
      } catch {
        continue;
      }
      if (res.status === "pending") continue;
      setMessages((ms) => ms.map((m) => (m.meta?.turn_id === turnId
        ? { ...m, meta: { ...m.meta, memory_pending: false, memory_ops: res.memory_ops } } : m)));
      const changed = res.memory_ops.filter((o) => o.op !== "NOOP" && o.memory_id).map((o) => o.memory_id!);
      if (changed.length) {
        setFlash(new Set(changed));
        setTimeout(() => setFlash(new Set()), 2500);
      }
      refresh();
      return;
    }
  }

  async function toggleMic() {
    if (recording) {
      setRecording(null);
      try {
        const text = await recording.stop();
        if (text) send(text);
      } catch (e) {
        alert(`Transcription failed: ${(e as Error).message}`);
      }
      return;
    }
    try {
      stopSpeaking();
      setRecording(await startRecording(voice.whisper));
    } catch (e) {
      alert((e as Error).message);
    }
  }

  function newSession() {
    stopSpeaking();
    const id = newSessionId();
    setSessionId(id);
    setMessages([]);
  }

  async function forget(id: string) {
    await api.deleteMemory(id);
    refresh();
  }

  async function forgetAll() {
    if (!confirm("Erase everything the assistant remembers about you?")) return;
    await api.clearMemories();
    refresh();
  }

  const pinned = memories.filter((m) => m.pinned);
  const byCategory = memories
    .filter((m) => !m.pinned)
    .reduce<Record<string, Memory[]>>((acc, m) => ((acc[m.category] ??= []).push(m), acc), {});

  return (
    <div className="app">
      <main className="chat">
        <header className="topbar">
          <div className="brand">
            <span className="logo" aria-hidden>◎</span>
            <div>
              <h1>Juno</h1>
              <p className="sub">Session {sessionId.slice(2)} · remembers across sessions</p>
            </div>
          </div>
          <div className="actions">
            <button className={`chip ${voiceOut ? "on" : ""}`} onClick={() => { setVoiceOut((v) => !v); stopSpeaking(); }}
              title="Read replies aloud">
              {voiceOut ? "🔊" : "🔈"}<span className="lbl"> {voiceOut ? "Voice on" : "Voice off"}</span>
            </button>
            <button className="chip" onClick={newSession} title="Start a new conversation — memory carries over">
              ＋<span className="lbl"> New session</span>
            </button>
            <button className="chip panel-toggle" onClick={() => setPanelOpen((v) => !v)}>
              🧠 {memories.length}
            </button>
          </div>
        </header>

        <section className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <h2>Tell me about yourself — I'll remember.</h2>
              <p>Try: “I live in Seattle and I prefer short answers.” Then start a new session and ask for a weekend plan.</p>
              <div className="suggestions">
                {["I'm a vegetarian and I'm training for a marathon.", "Actually, I just moved to Austin.",
                  "What do you know about me?", "Remind me to book flights on Friday."].map((s) => (
                  <button key={s} onClick={() => send(s)}>{s}</button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role} ${m.error ? "error" : ""}`}>
              <div className="bubble">{m.content}</div>
              {m.meta && <TurnMeta meta={m.meta} />}
            </div>
          ))}
          {busy && (
            <div className="msg assistant">
              <div className="bubble typing"><span /><span /><span /></div>
            </div>
          )}
          <div ref={endRef} />
        </section>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
          {sttSupported(voice.whisper) && (
            <button type="button" className={`mic ${recording ? "rec" : ""}`} onClick={toggleMic}
              title={recording ? "Stop and send" : voice.whisper ? "Speak (Whisper)" : "Speak (browser)"}>
              {recording ? "■" : "🎙"}
            </button>
          )}
          <input value={input} onChange={(e) => setInput(e.target.value)} disabled={busy}
            placeholder={recording ? "Listening… click ■ to send" : "Message Juno"} />
          <button type="submit" className="send" disabled={busy || !input.trim()}>Send</button>
        </form>
      </main>

      <aside className={`panel ${panelOpen ? "open" : ""}`}>
        <div className="panel-head">
          <h2>What I remember about you</h2>
          <div className="panel-head-actions">
            {memories.length > 0 && <button className="link" onClick={forgetAll}>Forget all</button>}
            <button className="x close" onClick={() => setPanelOpen(false)} aria-label="Close panel">×</button>
          </div>
        </div>

        {memories.length === 0 && <p className="muted">Nothing yet. Durable facts you mention will show up here.</p>}

        {pinned.length > 0 && (
          <div className="group">
            <h3>Always applied</h3>
            {pinned.map((m) => <MemoryRow key={m.id} m={m} flash={flash.has(m.id)} onForget={forget} />)}
          </div>
        )}
        {Object.entries(byCategory).map(([cat, items]) => (
          <div className="group" key={cat}>
            <h3>{cat}</h3>
            {items.map((m) => <MemoryRow key={m.id} m={m} flash={flash.has(m.id)} onForget={forget} />)}
          </div>
        ))}

        {events.length > 0 && (
          <div className="group">
            <h3>Recent memory changes</h3>
            <ul className="events">
              {events.slice(0, 12).map((e) => (
                <li key={e.id} className={`ev ${e.op.toLowerCase()}`}>
                  <span className="op">{e.op === "ADD" ? "+" : e.op === "UPDATE" ? "↻" : "−"}</span>
                  {e.op === "UPDATE" ? (
                    <span><s>{e.old_content}</s> → {e.new_content}</span>
                  ) : (
                    <span>{e.op === "DELETE" ? <s>{e.old_content}</s> : e.new_content}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {blocked.length > 0 && (
          <div className="group">
            <h3>Blocked from memory</h3>
            <ul className="events">
              {blocked.slice(0, 8).map((b) => (
                <li key={b.id} className="ev delete" title={b.reason ?? ""}>
                  <span className="op">⛔</span>
                  <span>{b.content} <em className="muted">({b.category.replace(/_/g, " ")})</em></span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="group">
          <h3>Tasks <span className="tag">MCP</span></h3>
          {tasks.length === 0 && <p className="muted">Say “remind me to…” to add one.</p>}
          {tasks.map((t) => (
            <label key={t.id} className={`task ${t.done ? "done" : ""}`}>
              <input type="checkbox" checked={t.done} disabled={t.done}
                onChange={() => api.completeTask(t.id).then(refresh)} />
              <span>{t.title}{t.due && <em> · {t.due}</em>}</span>
            </label>
          ))}
        </div>
      </aside>
    </div>
  );
}

function MemoryRow({ m, flash, onForget }: { m: Memory; flash: boolean; onForget: (id: string) => void }) {
  return (
    <div className={`memory ${flash ? "flash" : ""}`} title={`Updated ${new Date(m.updated_at).toLocaleString()} · used ${m.access_count}×`}>
      <span>{m.content.replace(/^User('s)?\s+/i, (_, s) => (s ? "Your " : ""))}</span>
      <button className="x" onClick={() => onForget(m.id)} aria-label="Forget this">×</button>
    </div>
  );
}

function TurnMeta({ meta }: { meta: Omit<ChatResponse, "reply"> }) {
  const [open, setOpen] = useState(false);
  const ops = meta.memory_ops.filter((o) => o.op !== "NOOP");
  const used = meta.used_memories;
  if (!ops.length && !used.length && !meta.tool_calls.length && !meta.memory_pending) return null;
  return (
    <div className="meta">
      {used.length > 0 && (
        <button className="pill used" onClick={() => setOpen((v) => !v)}>
          used {used.length} {used.length === 1 ? "memory" : "memories"} {open ? "▴" : "▾"}
        </button>
      )}
      {ops.map((o, i) => (
        <span key={i} className={`pill ${o.op.toLowerCase()}`} title={o.reason}>
          {o.op === "ADD" && <>+ remembered: {o.content}</>}
          {o.op === "UPDATE" && <>↻ updated: <s>{o.old_content}</s> → {o.content}</>}
          {o.op === "DELETE" && <>− forgot: {o.content}</>}
          {o.op === "QUARANTINE" && <>⛔ blocked from memory: {o.content}</>}
        </span>
      ))}
      {meta.memory_pending && <span className="pill pending">saving to memory…</span>}
      {meta.tool_calls.map((t, i) => (
        <span key={`t${i}`} className={`pill tool ${t.is_error ? "err" : ""}`} title={t.output}>
          ⚙ {t.name.replace(/_/g, " ")}
        </span>
      ))}
      {open && (
        <ul className="used-list">
          {used.map((u) => (
            <li key={u.id}>
              {u.content}
              <span className="muted">{u.pinned ? " · always" : u.similarity != null ? ` · ${u.similarity.toFixed(2)}` : ""}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
