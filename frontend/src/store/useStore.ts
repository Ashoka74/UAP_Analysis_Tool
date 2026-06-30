import { create } from 'zustand';
import type {
  PageId,
  DataResponse,
  ColumnStat,
  AnalysisResponse,
  DashboardSummary,
  CramersVResponse,
  ContingencyResponse,
  XGBoostResult,
  XgboostPcaResponse,
  XgboostImputeResponse,
} from '../types';

// Cramér's V explorer state — kept in the global store (not the component) so it
// survives unmounting when the user switches Analysis tabs / pages.
interface CramersParams {
  dropMissing: boolean;
  excludeTrivial: boolean;
  strong: number;
}
interface CramersContingency {
  pair: { a: string; b: string } | null;
  data: ContingencyResponse | null;
}

interface AppState {
  // Navigation
  currentPage: PageId;
  setPage: (page: PageId) => void;

  // Data
  dataLoaded: boolean;
  data: DataResponse | null;
  columnStats: ColumnStat[];
  setData: (data: DataResponse, stats: ColumnStat[]) => void;

  // Analysis
  analysisRunning: boolean;
  analysisResults: AnalysisResponse | null;
  setAnalysisRunning: (running: boolean) => void;
  setAnalysisResults: (results: AnalysisResponse) => void;

  // Cramér's V explorer (cached across tab/page switches)
  cramersSelected: string[] | null; // null = not yet initialized → defaults to all eligible
  cramersReport: CramersVResponse | null;
  cramersParams: CramersParams;
  cramersContingency: CramersContingency;
  cramersLocalXgb: Record<string, XGBoostResult> | null;
  cramersLocalPca: XgboostPcaResponse | null; // 2nd-pass payload for the inline render
  cramersLocalImpute: XgboostImputeResponse | null; // imputation payload for the inline render
  cramersAutoRun: boolean; // transient: compute once after a hand-off from another tab
  setCramersSelected: (cols: string[] | null) => void;
  setCramersReport: (report: CramersVResponse | null) => void;
  setCramersParams: (p: Partial<CramersParams>) => void;
  setCramersContingency: (pair: { a: string; b: string } | null, data: ContingencyResponse | null) => void;
  setCramersLocalXgb: (r: Record<string, XGBoostResult> | null) => void;
  setCramersLocalPca: (r: XgboostPcaResponse | null) => void;
  setCramersLocalImpute: (r: XgboostImputeResponse | null) => void;
  setCramersAutoRun: (v: boolean) => void;

  // Imputation result shared with the Data Explorer grid overlay (predicted
  // missing-value fills, coloured by model accuracy). Set whenever an imputation
  // run completes anywhere in the Analysis flow.
  imputation: XgboostImputeResponse | null;
  setImputation: (r: XgboostImputeResponse | null) => void;
  showImputed: boolean;            // Data Explorer overlay toggle
  setShowImputed: (v: boolean) => void;

  // Dashboard
  summary: DashboardSummary | null;
  setSummary: (summary: DashboardSummary) => void;

  // API keys
  geminiKey: string;
  setGeminiKey: (key: string) => void;
  openaiKey: string;
  setOpenaiKey: (key: string) => void;
  deepseekKey: string;
  setDeepseekKey: (key: string) => void;
  cohereKey: string;
  setCohereKey: (key: string) => void;
  // Neon connection string for the multimodal (pgvector) search
  dbUrl: string;
  setDbUrl: (url: string) => void;

  // Parsing → SCU handoff
  parsedReady: boolean;
  setParsedReady: (ready: boolean) => void;

  // Sidebar
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
}

export const useStore = create<AppState>((set) => ({
  currentPage: 'dashboard',
  setPage: (page) => set({ currentPage: page }),

  dataLoaded: false,
  data: null,
  columnStats: [],
  setData: (data, stats) => set({ data, columnStats: stats, dataLoaded: true }),

  analysisRunning: false,
  analysisResults: null,
  setAnalysisRunning: (running) => set({ analysisRunning: running }),
  setAnalysisResults: (results) => set({ analysisResults: results, analysisRunning: false }),

  cramersSelected: null,
  cramersReport: null,
  cramersParams: { dropMissing: false, excludeTrivial: true, strong: 0.3 },
  cramersContingency: { pair: null, data: null },
  cramersLocalXgb: null,
  cramersLocalPca: null,
  cramersLocalImpute: null,
  cramersAutoRun: false,
  setCramersSelected: (cols) => set({ cramersSelected: cols }),
  setCramersReport: (report) => set({ cramersReport: report }),
  setCramersParams: (p) => set((s) => ({ cramersParams: { ...s.cramersParams, ...p } })),
  setCramersContingency: (pair, data) => set({ cramersContingency: { pair, data } }),
  setCramersLocalXgb: (r) => set({ cramersLocalXgb: r }),
  setCramersLocalPca: (r) => set({ cramersLocalPca: r }),
  setCramersLocalImpute: (r) => set({ cramersLocalImpute: r }),
  setCramersAutoRun: (v) => set({ cramersAutoRun: v }),

  imputation: null,
  setImputation: (r) => set({ imputation: r }),
  showImputed: true,
  setShowImputed: (v) => set({ showImputed: v }),

  summary: null,
  setSummary: (summary) => set({ summary }),

  geminiKey: '',
  setGeminiKey: (key) => set({ geminiKey: key }),
  openaiKey: '',
  setOpenaiKey: (key) => set({ openaiKey: key }),
  deepseekKey: '',
  setDeepseekKey: (key) => set({ deepseekKey: key }),
  cohereKey: '',
  setCohereKey: (key) => set({ cohereKey: key }),
  dbUrl: '',
  setDbUrl: (url) => set({ dbUrl: url }),

  parsedReady: false,
  setParsedReady: (ready) => set({ parsedReady: ready }),

  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
