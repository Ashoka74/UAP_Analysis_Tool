import { Activity, Bell, Settings } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { StatusBadge } from '../common/StatusBadge';

const pageTitles: Record<string, string> = {
  dashboard: 'Mission Overview',
  data: 'Data Explorer',
  analysis: 'Pattern Analysis',
  query: 'AI Intelligence Query',
};

export function TopBar() {
  const { currentPage, dataLoaded, analysisRunning } = useStore();

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-abyss px-6">
      <div className="flex items-center gap-4">
        <h1 className="text-base font-semibold text-text-primary">
          {pageTitles[currentPage] ?? ''}
        </h1>
        {analysisRunning && <StatusBadge status="info" label="Processing" pulse />}
      </div>

      <div className="flex items-center gap-4">
        <StatusBadge
          status={dataLoaded ? 'success' : 'idle'}
          label={dataLoaded ? 'Data Loaded' : 'No Data'}
        />
        <button className="text-text-muted hover:text-text-secondary">
          <Activity className="h-4 w-4" />
        </button>
        <button className="text-text-muted hover:text-text-secondary">
          <Bell className="h-4 w-4" />
        </button>
        <button className="text-text-muted hover:text-text-secondary">
          <Settings className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}
