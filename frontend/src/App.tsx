import { AppShell } from './components/layout/AppShell';
import { Dashboard } from './components/dashboard/Dashboard';
import { DataExplorer } from './components/data/DataExplorer';
import { ParsingPage } from './components/parsing/ParsingPage';
import { AnalysisPage } from './components/analysis/AnalysisPage';
import { QueryPage } from './components/query/QueryPage';
import { RagSearchPage } from './components/rag/RagSearchPage';
import { ScuPage } from './components/scu/ScuPage';
import { MapPage } from './components/map/MapPage';
import { MagneticPage } from './components/magnetic/MagneticPage';
import { ClusterView } from './components/analysis/ClusterView';
import { DedupPage } from './components/dedup/DedupPage';
import { useStore } from './store/useStore';

function App() {
  const { currentPage } = useStore();

  return (
    <AppShell>
      {currentPage === 'dashboard' && <Dashboard />}
      {currentPage === 'data' && <DataExplorer />}
      {currentPage === 'parsing' && <ParsingPage />}
      {currentPage === 'analysis' && <AnalysisPage />}
      {currentPage === 'query' && <QueryPage />}
      {currentPage === 'rag' && <RagSearchPage />}
      {currentPage === 'scu' && <ScuPage />}
      {currentPage === 'map' && <MapPage />}
      {currentPage === 'magnetic' && <MagneticPage />}
      {currentPage === 'clusters' && <ClusterView />}
      {currentPage === 'dedup' && <DedupPage />}
    </AppShell>
  );
}

export default App;
