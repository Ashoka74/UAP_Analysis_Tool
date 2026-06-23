import { create } from 'zustand';
import type {
  PageId,
  DataResponse,
  ColumnStat,
  AnalysisResponse,
  DashboardSummary,
} from '../types';

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

  parsedReady: false,
  setParsedReady: (ready) => set({ parsedReady: ready }),

  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
