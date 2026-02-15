import { AppShell } from './components/layout/AppShell';
import { Dashboard } from './components/dashboard/Dashboard';
import { DataExplorer } from './components/data/DataExplorer';
import { AnalysisPage } from './components/analysis/AnalysisPage';
import { QueryPage } from './components/query/QueryPage';
import { useStore } from './store/useStore';

function App() {
  const { currentPage } = useStore();

  return (
    <AppShell>
      {currentPage === 'dashboard' && <Dashboard />}
      {currentPage === 'data' && <DataExplorer />}
      {currentPage === 'analysis' && <AnalysisPage />}
      {currentPage === 'query' && <QueryPage />}
    </AppShell>
  );
}

export default App;
