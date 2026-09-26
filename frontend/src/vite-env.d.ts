/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MOCK_WS?: string;
  readonly VITE_WS_PATH?: string;
  readonly VITE_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
