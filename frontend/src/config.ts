const env = import.meta.env;

/** Use mock predictions until the real backend (Agent 6) is wired. */
export const MOCK_WEBSOCKET =
  env.VITE_MOCK_WS !== "false" && env.VITE_MOCK_WS !== "0";

export const WEBSOCKET_PATH =
  (env.VITE_WS_PATH as string | undefined) ?? "/ws/recognize";

export function resolveWebSocketUrl(): string {
  if (env.VITE_WS_URL) {
    return env.VITE_WS_URL as string;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${WEBSOCKET_PATH}`;
}

export const INFERENCE = {
  confidenceThreshold: 0.6,
  debounceCooldownMs: 800,
} as const;

export const MOCK_VOCAB = [
  "hello",
  "thank you",
  "yes",
  "no",
  "help",
  "water",
  "friend",
  "family",
  "good",
  "bad",
] as const;

/** Placeholder Bangla glosses for mock word-by-word display. */
export const MOCK_BANGLA_GLOSS: Record<string, string> = {
  hello: "হ্যালো",
  "thank you": "ধন্যবাদ",
  yes: "হ্যাঁ",
  no: "না",
  help: "সাহায্য",
  water: "পানি",
  friend: "বন্ধু",
  family: "পরিবার",
  good: "ভালো",
  bad: "খারাপ",
};
