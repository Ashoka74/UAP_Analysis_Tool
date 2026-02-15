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

  // Gemini
  geminiKey: string;
  setGeminiKey: (key: string) => void;

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

  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
