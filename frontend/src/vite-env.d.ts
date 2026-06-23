/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend API origin for split deploys, e.g. "https://user-uap.hf.space". */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
