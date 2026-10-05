export type Role = "user" | "assistant";

export interface Memory {
  id: string;
  content: string;
  category: string;
  pinned: boolean;
  confidence: number;
  created_at: string;
  updated_at: string;
  access_count: number;
}

export interface MemoryEvent {
  id: number;
  memory_id: string | null;
  op: "ADD" | "UPDATE" | "DELETE";
  old_content: string | null;
  new_content: string | null;
  reason: string | null;
  created_at: string;
}

export interface MemoryOp {
  op: "ADD" | "UPDATE" | "DELETE" | "NOOP";
  content: string;
  memory_id: string | null;
  old_content: string | null;
  reason: string;
  path: string;
}

export interface UsedMemory {
  id: string;
  content: string;
  pinned: boolean;
  similarity?: number | null;
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output: string;
  is_error: boolean;
}

export interface ChatResponse {
  reply: string;
  route: { needs_memory: boolean; search_queries: string[]; may_contain_facts: boolean } | null;
  used_memories: UsedMemory[];
  memory_ops: MemoryOp[];
  memory_pending: boolean;
  turn_id: string | null;
  tool_calls: ToolCall[];
}

export interface Task {
  id: number;
  title: string;
  due: string | null;
  done: boolean;
  created_at: string;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      /* not JSON */
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  chat: (session_id: string, message: string,
         history: { role: Role; content: string; tool_calls?: Pick<ToolCall, "name" | "input" | "output">[] }[]) =>
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, message, history }),
    }).then((r) => json<ChatResponse>(r)),
  memories: () => fetch("/api/memories").then((r) => json<Memory[]>(r)),
  events: () => fetch("/api/memories/events?limit=30").then((r) => json<MemoryEvent[]>(r)),
  deleteMemory: (id: string) => fetch(`/api/memories/${id}`, { method: "DELETE" }).then((r) => json(r)),
  clearMemories: () => fetch("/api/memories", { method: "DELETE" }).then((r) => json(r)),
  tasks: () => fetch("/api/tasks").then((r) => json<Task[]>(r)),
  completeTask: (id: number) => fetch(`/api/tasks/${id}/done`, { method: "POST" }).then((r) => json(r)),
  voiceStatus: () => fetch("/api/voice/status").then((r) => json<{ whisper: boolean; elevenlabs: boolean }>(r)),
  transcribe: (blob: Blob) => {
    const fd = new FormData();
    fd.append("audio", blob, "speech.webm");
    return fetch("/api/transcribe", { method: "POST", body: fd }).then((r) => json<{ text: string }>(r));
  },
  turnMemory: (turnId: string) =>
    fetch(`/api/turns/${turnId}`).then((r) =>
      json<{ status: "pending" | "done" | "error"; memory_ops: MemoryOp[] }>(r)),
  tts: (text: string) =>
    fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then((r) => {
      if (!r.ok) throw new Error(`tts ${r.status}`);
      return r.blob();
    }),
};
