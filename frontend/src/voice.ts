import { api } from "./api";

/* ---------- speech-to-text ---------- */

// Server Whisper when available (record -> upload), otherwise the browser's
// built-in SpeechRecognition (Chrome/Edge/Safari).
type SR = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
};

function browserRecognizer(): SR | null {
  const w = window as unknown as { SpeechRecognition?: new () => SR; webkitSpeechRecognition?: new () => SR };
  const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition;
  return Ctor ? new Ctor() : null;
}

export function sttSupported(whisper: boolean): boolean {
  return (whisper && !!navigator.mediaDevices?.getUserMedia) || browserRecognizer() !== null;
}

export interface Recording {
  stop(): Promise<string>;
}

export async function startRecording(useWhisper: boolean): Promise<Recording> {
  if (useWhisper && navigator.mediaDevices?.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.start();
    return {
      stop: () =>
        new Promise<string>((resolve, reject) => {
          rec.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop());
            try {
              const { text } = await api.transcribe(new Blob(chunks, { type: rec.mimeType }));
              resolve(text);
            } catch (e) {
              reject(e);
            }
          };
          rec.stop();
        }),
    };
  }

  const sr = browserRecognizer();
  if (!sr) throw new Error("Speech recognition isn't supported in this browser.");
  sr.lang = "en-US";
  sr.interimResults = false;
  sr.continuous = true;
  let text = "";
  let done: ((t: string) => void) | null = null;
  sr.onresult = (e) => {
    text = Array.from(e.results).map((r) => r[0].transcript).join(" ");
  };
  sr.onerror = () => done?.(text);
  sr.onend = () => done?.(text);
  sr.start();
  return {
    stop: () =>
      new Promise<string>((resolve) => {
        done = resolve;
        sr.stop();
      }),
  };
}

/* ---------- text-to-speech ---------- */

let current: HTMLAudioElement | null = null;

export function stopSpeaking() {
  current?.pause();
  current = null;
  window.speechSynthesis?.cancel();
}

function plain(text: string) {
  return text.replace(/[*_`#>]/g, "").replace(/\[(.*?)\]\(.*?\)/g, "$1");
}

export async function speak(text: string, useElevenLabs: boolean) {
  stopSpeaking();
  const t = plain(text);
  if (useElevenLabs) {
    try {
      const blob = await api.tts(t);
      current = new Audio(URL.createObjectURL(blob));
      await current.play();
      return;
    } catch {
      /* fall through to the browser voice */
    }
  }
  if ("speechSynthesis" in window) {
    const u = new SpeechSynthesisUtterance(t);
    u.rate = 1.05;
    window.speechSynthesis.speak(u);
  }
}
