export type TopKCandidate = {
  word: string;
  confidence: number;
};

export type PredictionMessage = {
  type: "prediction";
  word: string;
  confidence: number;
  top_k: TopKCandidate[];
  model_version?: string;
  vocab_size?: number;
  latency_ms?: number;
  committed_word?: string | null;
};

export type StatusMessage = {
  type: "status";
  connection: "connected" | "disconnected" | "error";
  message?: string;
};

export type ServerMessage = PredictionMessage | StatusMessage;

export type LandmarkFrame = {
  timestamp_ms: number;
  landmarks: number[];
};

export type LandmarkWindowMessage = {
  type: "landmark_window";
  frames: LandmarkFrame[];
  session_id?: string;
};

export type ClientMessage = LandmarkWindowMessage;

export type ConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";
