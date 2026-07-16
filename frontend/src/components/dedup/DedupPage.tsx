import { useState, useEffect } from 'react';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Play,
  Sparkles,
  Layers,
  Database,
  FileText,
  Calendar,
  MapPin,
  Puzzle,
  Zap,
  Target,
  Globe,
  Download,
  Upload,
} from 'lucide-react';
import type {
  AdvancedDedupResponse,
  CrossDbPipelineResponse,
  CrossDbPair,
} from '../../types';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Panel } from '../common/Panel';

function DedupeProgressBar({ loading, title }: { loading: boolean; title?: string }) {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');

  useEffect(() => {
    if (!loading) {
      setProgress(100);
      return;
    }
    setProgress(8);
    setStage('Stage 1/4: Extracting & normalizing report metadata & composite text features...');

    const t1 = setTimeout(() => {
      setProgress(32);
      setStage('Stage 2/4: Computing dense vector embeddings using Harrier (microsoft/harrier-oss-v1-270m)...');
    }, 700);

    const t2 = setTimeout(() => {
      setProgress(64);
      setStage('Stage 3/4: Evaluating pairwise cosine similarity & screening multi-gate spatial/temporal criteria...');
    }, 1800);

    const t3 = setTimeout(() => {
      setProgress(86);
      setStage('Stage 4/4: Constructing transitive duplicate clustering graph & allocating similarity bins...');
    }, 3500);

    const t4 = setTimeout(() => {
      setProgress(95);
      setStage('Finalizing batch matrix evaluation across candidate pairs...');
    }, 5500);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
    };
  }, [loading]);

  if (!loading) return null;

  return (
    <div className="mt-3 flex flex-col gap-2 rounded-md border border-border bg-raised p-3">
      <div className="flex items-center justify-between text-xs font-medium">
        <span className="flex items-center gap-2 text-accent">
          <Sparkles className="h-3.5 w-3.5 animate-spin" /> {title || 'Running Harrier Deduplication Pipeline...'}
        </span>
        <span className="font-mono text-text-secondary">{progress}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-deep">
        <div
          className="h-full rounded-full bg-accent transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      <span className="truncate font-mono text-[11px] text-text-muted">{stage}</span>
    </div>
  );
}

// Full side-by-side comparison for one candidate pair — shared by the
// Cross-DB tab's Preview and the Batch Cluster tab's Stage A audit trail
// preview, since both operate on the same CrossDbPair shape.
function AlignedInspector({ pair }: { pair: CrossDbPair }) {
  return (
    <div className="flex flex-col">
      {/* Inspector Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div className="flex flex-col">
          <span className="flex items-center gap-2 text-sm font-medium text-text-primary">
            <Database className="h-4 w-4 text-text-muted" /> Pair #{pair.id_a} vs #{pair.id_b}
          </span>
          <span className="mt-0.5 text-xs text-text-muted">
            Record A and Record B, aligned side-by-side by database column
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="rounded border border-border bg-deep px-2.5 py-1 font-mono text-xs text-text-secondary">
            Sim: {(pair.similarity * 100).toFixed(1)}%
          </span>
          {pair.haversine_km !== null && (
            <span className="rounded border border-border bg-deep px-2.5 py-1 font-mono text-xs text-text-secondary">
              Dist: {pair.haversine_km} km
            </span>
          )}
          {pair.date_diff_days !== null && (
            <span className="rounded border border-border bg-deep px-2.5 py-1 font-mono text-xs text-text-secondary">
              Time Gap: {pair.date_diff_days} days
            </span>
          )}
        </div>
      </div>

      {/* Aligned Side-by-Side Comparison Table */}
      <div className="max-h-[580px] overflow-x-auto overflow-y-auto">
        <table className="w-full border-collapse text-left text-xs">
          <thead>
            <tr className="sticky top-0 border-b border-border bg-deep font-mono text-text-secondary">
              <th className="w-1/4 p-3.5">Database Attribute</th>
              <th className="w-3/8 border-l border-border/50 p-3.5 text-accent">
                Row A (#{pair.id_a})
              </th>
              <th className="w-3/8 border-l border-border/50 p-3.5 text-text-primary">
                Row B (#{pair.id_b})
              </th>
              <th className="w-1/6 border-l border-border/50 p-3.5">Gate Alignment</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/40 text-text-primary">
            {/* Record ID Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 font-mono font-medium text-text-secondary">Primary Record ID</td>
              <td className="border-l border-border/50 p-3.5 font-mono font-medium text-accent">{pair.id_a}</td>
              <td className="border-l border-border/50 p-3.5 font-mono font-medium text-text-primary">{pair.id_b}</td>
              <td className="border-l border-border/50 p-3.5 font-mono text-text-muted">Candidate Pair</td>
            </tr>

            {/* Witness Notes / Narrative Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 align-top font-medium text-text-primary">
                Witness Notes / Narrative Text
              </td>
              <td className="border-l border-border/50 p-3.5 align-top leading-relaxed text-text-primary">
                {pair.row_a?.witness_notes || pair.row_a?.narrative || pair.text_a_preview || '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top leading-relaxed text-text-primary">
                {pair.row_b?.witness_notes || pair.row_b?.narrative || pair.text_b_preview || '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top">
                <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                  pair.flags.is_similar_text ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                }`}>
                  <span className="flex items-center gap-1">
                    {pair.flags.is_similar_text ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                    {pair.flags.is_similar_text ? 'GATE PASSED' : 'BELOW THRESHOLD'}
                  </span>
                  <span className="text-[10px] font-normal">{(pair.similarity * 100).toFixed(1)}% Cosine</span>
                </span>
              </td>
            </tr>

            {/* Date & Time Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 align-top font-medium text-text-secondary">Timestamp / Date</td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {String(pair.row_a?.date_time || pair.row_a?.date || '—')}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {String(pair.row_b?.date_time || pair.row_b?.date || '—')}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top">
                {pair.date_diff_days !== null ? (
                  <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                    pair.flags.is_similar_date ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                  }`}>
                    <span className="flex items-center gap-1">
                      {pair.flags.is_similar_date ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                      {pair.flags.is_similar_date ? 'GATE PASSED' : 'OUT OF RANGE'}
                    </span>
                    <span className="text-[10px] font-normal">Δ {pair.date_diff_days} days</span>
                  </span>
                ) : (
                  <span className="block text-center font-mono text-text-muted">No Date Data</span>
                )}
              </td>
            </tr>

            {/* Coordinates Row */}
            <tr className="hover:bg-elevated/30 transition-colors">
              <td className="bg-deep/60 p-3.5 align-top font-medium text-text-secondary">Spatial Coordinates</td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {pair.row_a?.latitude !== undefined ? `${Number(pair.row_a.latitude).toFixed(4)}, ${Number(pair.row_a.longitude).toFixed(4)}` : '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                {pair.row_b?.latitude !== undefined ? `${Number(pair.row_b.latitude).toFixed(4)}, ${Number(pair.row_b.longitude).toFixed(4)}` : '—'}
              </td>
              <td className="border-l border-border/50 p-3.5 align-top">
                {pair.haversine_km !== null ? (
                  <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                    pair.flags.is_similar_location ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                  }`}>
                    <span className="flex items-center gap-1">
                      {pair.flags.is_similar_location ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                      {pair.flags.is_similar_location ? 'GATE PASSED' : 'OUT OF RANGE'}
                    </span>
                    <span className="text-[10px] font-normal">Δ {pair.haversine_km} km</span>
                  </span>
                ) : pair.location_name_similarity !== null ? (
                  <span className={`flex flex-col items-center gap-0.5 rounded border px-2 py-1 text-center font-mono text-xs font-medium ${
                    pair.flags.is_similar_location ? 'border-success/30 bg-success/10 text-success' : 'border-border bg-elevated text-text-muted'
                  }`}>
                    <span className="flex items-center gap-1">
                      {pair.flags.is_similar_location ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                      {pair.flags.is_similar_location ? 'GATE PASSED' : 'BELOW THRESHOLD'}
                    </span>
                    <span className="text-[10px] font-normal">Name sim: {(pair.location_name_similarity * 100).toFixed(0)}%</span>
                  </span>
                ) : (
                  <span className="block text-center font-mono text-text-muted">No Lat/Lon</span>
                )}
              </td>
            </tr>

            {/* Additional Aligned Database Columns */}
            {(() => {
              const rowAKeys = pair.row_a ? Object.keys(pair.row_a) : [];
              const rowBKeys = pair.row_b ? Object.keys(pair.row_b) : [];
              const ignoreKeys = ['id', 'locus_tag', 'case_id', 'witness_notes', 'narrative', 'description', 'summary', 'text', 'date_time', 'date', 'datetime', 'latitude', 'lat', 'longitude', 'lon', 'lng'];
              const allOtherKeys = Array.from(new Set([...rowAKeys, ...rowBKeys])).filter(k => !ignoreKeys.includes(k.toLowerCase()));

              return allOtherKeys.map((key) => {
                const valA = pair.row_a?.[key] !== undefined ? String(pair.row_a[key]) : '—';
                const valB = pair.row_b?.[key] !== undefined ? String(pair.row_b[key]) : '—';
                const isExact = valA !== '—' && valB !== '—' && valA.toLowerCase() === valB.toLowerCase();

                return (
                  <tr key={key} className="hover:bg-elevated/30 transition-colors">
                    <td className="bg-deep/60 p-3.5 align-top font-medium capitalize text-text-secondary">
                      {key.replace(/_/g, ' ')}
                    </td>
                    <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                      {valA}
                    </td>
                    <td className="border-l border-border/50 p-3.5 align-top font-mono text-text-primary">
                      {valB}
                    </td>
                    <td className="border-l border-border/50 p-3.5 align-top">
                      {isExact ? (
                        <span className="flex items-center justify-center gap-1 rounded border border-success/30 bg-success/10 px-2 py-0.5 text-center font-mono text-[11px] font-medium text-success">
                          <CheckCircle2 className="h-3 w-3" /> Exact Match
                        </span>
                      ) : (
                        <span className="block text-center font-mono text-[11px] text-text-muted">
                          Differs
                        </span>
                      )}
                    </td>
                  </tr>
                );
              });
            })()}

            {/* Final Multi-Gate Verdict Row */}
            <tr className="border-t-2 border-border bg-deep font-medium">
              <td className="bg-elevated/40 p-4 text-text-primary">Full Multi-Gate Verdict</td>
              <td colSpan={2} className="border-l border-border/50 p-4 text-text-primary">
                {pair.flags.is_similar_all ? (
                  <span className="flex items-center gap-2 text-success">
                    <CheckCircle2 className="h-4 w-4" /> Candidate Duplicate — text + spatial + temporal convergence
                  </span>
                ) : pair.flags.is_similar_both ? (
                  <span className="flex items-center gap-2 text-warning">
                    <Zap className="h-4 w-4" /> Strong spatial + temporal correlation (distinct narrative)
                  </span>
                ) : (
                  <span className="text-text-secondary">
                    Partial match — classified as {pair.bin.replace(/_/g, ' ')}
                  </span>
                )}
              </td>
              <td className="border-l border-border/50 p-4 text-center">
                <span className={`rounded-full border px-3 py-1 font-mono text-xs font-medium ${
                  pair.flags.is_similar_all ? 'border-success/40 bg-success/20 text-success' : 'border-border bg-elevated text-text-secondary'
                }`}>
                  {pair.flags.is_similar_all ? 'DUPLICATE' : 'DISTINCT'}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Stage A audit-trail sort priority: Full Convergence > Date+Location >
// Text+Location > Text+Date > Text-only > everything else. Checked in this
// order rather than independently, since the flags overlap — is_similar_all
// already implies is_similar_both, so it has to be checked first to keep
// the buckets mutually exclusive.
function gateSortRank(p: CrossDbPair): number {
  if (p.flags.is_similar_all) return 0;
  if (p.flags.is_similar_both) return 1;
  if (p.flags.is_similar_text && p.flags.is_similar_location) return 2;
  if (p.flags.is_similar_text_date) return 3;
  if (p.flags.is_similar_text) return 4;
  return 5;
}

export function DedupPage() {
  const { data, dataLoaded, setData, setPage } = useStore();
  const [activeTab, setActiveTab] = useState<'advanced' | 'cross_db'>('advanced');

  // Embedding column selection — shared by the Advanced (Batch Cluster) and
  // Cross-DB tabs, mirroring the Streamlit page-level column picker.
  const [embedCols, setEmbedCols] = useState<string[]>([]);
  const [embedColFilter, setEmbedColFilter] = useState('');
  const NARRATIVE_COL_HINTS = ['witness_notes', 'narrative', 'description', 'case_text.text', 'summary', 'text'];

  useEffect(() => {
    if (dataLoaded && data?.columns?.length) {
      const guessed = NARRATIVE_COL_HINTS.filter((c) => data.columns.includes(c));
      setEmbedCols(guessed.length ? guessed : [data.columns[0]]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataLoaded, data]);

  const toggleEmbedCol = (col: string) => {
    setEmbedCols((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  };

  const activeEmbedCols = embedCols.length > 0 ? embedCols : ['witness_notes'];

  // Field mapping (date / location) — same reasoning as the embedding column
  // picker above: the pipelines used to look for fixed column names
  // (date_time/date, latitude/lat, longitude/lon) and silently never fire
  // the date/location gates if a dataset used different names. Auto-detected
  // defaults below keep existing datasets working with zero clicks; leaving
  // these unset (a dataset that matches no hint) falls back to the
  // backend's own default-name guessing.
  const DATE_COL_HINTS = ['date_time', 'date', 'sighting_date', 'sightingDetails.date'];
  const LAT_COL_HINTS = ['latitude', 'lat', 'sightingDetails.location.latitude'];
  const LON_COL_HINTS = ['longitude', 'lon', 'lng', 'sightingDetails.location.longitude'];
  const LOCATION_NAME_HINTS = ['location.name', 'sightingDetails.location.name', 'location_name', 'city', 'city_name', 'location.city'];
  const STATE_COL_HINTS = ['location.state', 'location_state_norm', 'state', 'sightingDetails.location.state'];

  const [dateCol, setDateCol] = useState<string>('');
  const [locationMode, setLocationMode] = useState<'coords' | 'name'>('coords');
  const [latCol, setLatCol] = useState<string>('');
  const [lonCol, setLonCol] = useState<string>('');
  const [locationCol, setLocationCol] = useState<string>('');
  const [stateCol, setStateCol] = useState<string>('');

  useEffect(() => {
    if (dataLoaded && data?.columns?.length) {
      const guessedDate = DATE_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedLat = LAT_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedLon = LON_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedLoc = LOCATION_NAME_HINTS.find((c) => data.columns.includes(c)) ?? '';
      const guessedState = STATE_COL_HINTS.find((c) => data.columns.includes(c)) ?? '';
      setDateCol(guessedDate);
      setLatCol(guessedLat);
      setLonCol(guessedLon);
      setLocationCol(guessedLoc);
      setStateCol(guessedState);
      setLocationMode(guessedLat && guessedLon ? 'coords' : guessedLoc ? 'name' : 'coords');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataLoaded, data]);

  const activeDateCol = dateCol ? dateCol : undefined;
  const activeLatCol = locationMode === 'coords' && latCol ? latCol : undefined;
  const activeLonCol = locationMode === 'coords' && lonCol ? lonCol : undefined;
  const activeLocationCol = locationMode === 'name' && locationCol ? locationCol : undefined;
  const activeStateCol = locationMode === 'name' && stateCol ? stateCol : undefined;



  const [error, setError] = useState<string | null>(null);

  // Advanced Tab States
  const [threshold, setThreshold] = useState(0.80);
  const [dateDiffDays, setDateDiffDays] = useState(3);
  const [maxKm, setMaxKm] = useState(50.0);
  // Shared by both Batch Cluster and Cross-DB: cluster-block the pairwise
  // search (UMAP+HDBSCAN over row embeddings) instead of a dense N x N
  // similarity matrix. Off by default — only worth it once a dataset is
  // large enough that the full matrix stops being cheap (tens of thousands
  // of rows); see the block_stats note next to the toggle.
  const [useClusterBlocking, setUseClusterBlocking] = useState(false);
  const [blockMinClusterSize, setBlockMinClusterSize] = useState(15);
  // Opt-in like cluster-blocking: off by default because the gazetteer is
  // US-only and a bare place name can collide across states/countries.
  // When off, the location gate falls straight to text-similarity on the
  // Location Name Column instead of resolving a real distance.
  const [useGazetteer, setUseGazetteer] = useState(false);
  const [advResult, setAdvResult] = useState<AdvancedDedupResponse | null>(null);
  const [advLoading, setAdvLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const [applyLoading, setApplyLoading] = useState(false);
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [selectedAuditPairIndex, setSelectedAuditPairIndex] = useState<number>(0);
  const [auditViewMode, setAuditViewMode] = useState<'table' | 'preview'>('table');
  // Same bin/gate filter pattern as the Cross-DB tab, applied to Stage A's
  // own pairs array.
  const [auditBinFilter, setAuditBinFilter] = useState<string>('All');
  const [auditGateFilter, setAuditGateFilter] = useState<string>('show_all');

  // Cross-DB Pipeline States — bins and gates are both always-available
  // filters over one shared pairs table (no more separate Easy/Hard "path"
  // modes); crossViewMode switches that table for the Aligned Inspector.
  const [crossCmpMode, setCrossCmpMode] = useState<'within' | 'between'>('within');
  const [crossBinFilter, setCrossBinFilter] = useState<string>('All');
  const [crossGateFilter, setCrossGateFilter] = useState<string>('show_all');
  const [crossResult, setCrossResult] = useState<CrossDbPipelineResponse | null>(null);
  const [crossLoading, setCrossLoading] = useState(false);
  const [selectedPairIndex, setSelectedPairIndex] = useState<number>(0);
  const [crossViewMode, setCrossViewMode] = useState<'table' | 'preview'>('table');

  // Builds the active-dataset record list shared by both the Advanced (Batch
  // Cluster) and Cross-DB tabs — the loaded dataset, mapped so id/date/lat/
  // lon are normalized (narrative text comes from whichever columns are
  // selected via activeEmbedCols, so we don't synthesize a text field
  // here). Every loaded row is sent — no implicit truncation (matches
  // run_cross_db_pipeline's "no cap unless asked" default) — so for large
  // datasets, turn on Cluster-Blocked Comparison above instead of silently
  // comparing fewer rows than loaded.
  const buildDedupRecords = (): any[] => {
    if (!dataLoaded || !data?.rows) return [];
    return data.rows.map((row: any, idx: number) => {
      const id = row.id || row.locus_tag || row.case_id || `Case-${idx + 1}`;
      // Missing date/lat/lon must stay missing, not fall back to a fake
      // value — '2000-01-01' or (0,0) would make every undated/unlocated
      // row look identical to every other one, silently passing the
      // date/location similarity gates for pairs with no real evidence.
      const dt = row.date_time || row.date || row.datetime;
      const rawLat = row.latitude ?? row.lat;
      const rawLon = row.longitude ?? row.lon ?? row.lng;
      const lat = rawLat != null && rawLat !== '' ? Number(rawLat) : null;
      const lon = rawLon != null && rawLon !== '' ? Number(rawLon) : null;
      return {
        ...row,
        id: String(id),
        date_time: dt ? String(dt) : '',
        latitude: lat != null && !Number.isNaN(lat) ? lat : null,
        longitude: lon != null && !Number.isNaN(lon) ? lon : null,
      };
    });
  };

  const handleRunCrossDb = async () => {
    setCrossLoading(true);
    setError(null);
    try {
      const baseRecords = buildDedupRecords();
      let recordsA: any[] = baseRecords;
      let recordsB: any[] | null = null;
      if (crossCmpMode === 'between') {
        const half = Math.floor(baseRecords.length / 2);
        recordsA = baseRecords.slice(0, half);
        recordsB = baseRecords.slice(half);
      }

      const dataRes = await api.runCrossDbPipeline({
        records_a: recordsA,
        records_b: recordsB,
        cols_a: activeEmbedCols,
        cols_b: activeEmbedCols,
        threshold: threshold,
        max_days: dateDiffDays,
        max_km: maxKm,
        date_col: activeDateCol,
        lat_col: activeLatCol,
        lon_col: activeLonCol,
        location_col: activeLocationCol,
        state_col: activeStateCol,
        use_gazetteer: useGazetteer,
        use_cluster_blocking: useClusterBlocking,
        block_min_cluster_size: blockMinClusterSize,
      });
      setCrossResult(dataRes as CrossDbPipelineResponse);
      setSelectedPairIndex(0);
    } catch (err: any) {
      setError(err.message || 'Cross-DB pipeline evaluation failed. Check backend connection and dataset schema.');
    } finally {
      setCrossLoading(false);
    }
  };

  // API Call: Run Advanced Dedup (live multi-gate screening + union-find clustering)
  const handleRunAdvanced = async () => {
    setAdvLoading(true);
    setError(null);
    try {
      const dataRes = await api.runAdvancedDedup({
        records: buildDedupRecords(),
        cols: activeEmbedCols,
        threshold,
        date_diff_days: dateDiffDays,
        max_km: maxKm,
        use_llm_judge: false,
        date_col: activeDateCol,
        lat_col: activeLatCol,
        lon_col: activeLonCol,
        location_col: activeLocationCol,
        state_col: activeStateCol,
        use_gazetteer: useGazetteer,
        use_cluster_blocking: useClusterBlocking,
        block_min_cluster_size: blockMinClusterSize,
      });
      setAdvResult(dataRes as AdvancedDedupResponse);
    } catch (err: any) {
      setError(err.message || 'Advanced deduplication failed.');
    } finally {
      setAdvLoading(false);
    }
  };

  // Downloads deduped_dataset.csv (dedup_cluster_id/size/is_canonical/
  // is_duplicate baked onto every row from buildDedupRecords(), matching
  // exactly what Stage A/B were run on) + dedup_run_metadata.json, zipped —
  // the metadata records every parameter this run used, for reproducibility.
  const handleExportDedup = async () => {
    if (!advResult?.clusters) return;
    setExportLoading(true);
    setError(null);
    try {
      await api.exportDedupResults({
        records: buildDedupRecords(),
        clusters: advResult.clusters as unknown as Record<string, unknown>[],
        id_field: 'id',
        parameters: {
          threshold,
          date_diff_days: dateDiffDays,
          max_km: maxKm,
          embed_cols: activeEmbedCols,
          date_col: activeDateCol ?? null,
          lat_col: activeLatCol ?? null,
          lon_col: activeLonCol ?? null,
          location_col: activeLocationCol ?? null,
          state_col: activeStateCol ?? null,
          use_gazetteer: useGazetteer,
          use_cluster_blocking: useClusterBlocking,
          block_min_cluster_size: blockMinClusterSize,
          model_name: 'microsoft/harrier-oss-v1-270m',
          top_k: 5,
        },
      });
    } catch (err: any) {
      setError(err.message || 'Dedup export failed.');
    } finally {
      setExportLoading(false);
    }
  };

  // Same enrichment as the export, but loads the result straight into the
  // Data Explorer store instead of downloading a file — no download/
  // re-upload round trip needed to filter/sort on the new dedup_* columns.
  const handleApplyDedup = async () => {
    if (!advResult?.clusters) return;
    setApplyLoading(true);
    setError(null);
    try {
      const res = await api.applyDedupToDataset({
        records: buildDedupRecords(),
        clusters: advResult.clusters as unknown as Record<string, unknown>[],
        id_field: 'id',
      });
      setData(res.data, res.column_stats);
    } catch (err: any) {
      setError(err.message || 'Applying dedup results to the dataset failed.');
    } finally {
      setApplyLoading(false);
    }
  };


  const tabs: { id: 'advanced' | 'cross_db'; label: string; icon: typeof Layers }[] = [
    { id: 'advanced', label: 'Batch Cluster Pipeline', icon: Layers },
    // Hidden for now — Stage A's bins/gate filters + preview now give the
    // same audit capability Cross-DB's "within" mode offered, so it's
    // redundant with Batch Cluster Pipeline. Its one non-redundant job,
    // comparing two separate databases, is still there in the code
    // (crossCmpMode === 'between'); re-add this tab once that's the only
    // mode it exposes.
    // { id: 'cross_db', label: 'Cross-DB Pipeline', icon: Database },
  ];

  if (!dataLoaded) {
    return (
      <Panel title="No Data Loaded">
        <p className="text-sm text-text-muted">
          Load a dataset first from the{' '}
          <button onClick={() => setPage('dashboard')} className="text-accent hover:underline">
            Dashboard
          </button>{' '}
          or{' '}
          <button onClick={() => setPage('data')} className="text-accent hover:underline">
            Data Explorer
          </button>.
        </p>
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      {/* Tab bar — page identity ("Dedupe Studio") already shown in the TopBar */}
      <div className="flex flex-wrap gap-1 border-b border-border">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-1.5 border-b-2 px-4 py-2.5 text-xs font-medium transition-colors ${
              activeTab === id
                ? 'border-accent text-accent'
                : 'border-transparent text-text-muted hover:text-text-secondary'
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* SHARED: Data Source + Embedding Column Selection — feeds both the
          Advanced (Batch Cluster) and Cross-DB pipelines */}
      {(activeTab === 'advanced' || activeTab === 'cross_db') && (
        <Panel title="Data Source & Embedding Columns" subtitle="Feeds the Batch Cluster and Cross-DB pipelines below">
          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <Database className="h-3.5 w-3.5 text-text-muted" />
            Active dataset: {data?.rows ? `${data.rows.length.toLocaleString()} rows` : '0 rows'}
          </div>

          <div className="mt-3 flex flex-col gap-2 border-t border-border/50 pt-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
                <Layers className="h-3.5 w-3.5" /> Embedding Columns (Semantic Representation)
              </span>
              <span className="font-mono text-[10px] text-text-muted">
                {activeEmbedCols.length} selected
              </span>
            </div>
            {data?.columns?.length ? (
              <>
                <input
                  type="text"
                  value={embedColFilter}
                  onChange={(e) => setEmbedColFilter(e.target.value)}
                  placeholder="Filter columns..."
                  className="w-full rounded-md border border-border bg-deep px-3 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                />
                <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto p-1">
                  {data.columns
                    .filter((c) => c.toLowerCase().includes(embedColFilter.toLowerCase()))
                    .map((c) => (
                      <button
                        key={c}
                        onClick={() => toggleEmbedCol(c)}
                        className={`rounded border px-2 py-1 font-mono text-[11px] transition-colors ${
                          embedCols.includes(c)
                            ? 'border-accent bg-accent-dim/30 text-accent-bright'
                            : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        {c}
                      </button>
                    ))}
                </div>
                <p className="text-[10px] text-text-muted">
                  Selected columns are concatenated per row to build the narrative text used for semantic similarity.
                </p>
              </>
            ) : (
              <p className="text-[11px] text-text-muted">
                Load a dataset to pick which columns feed the narrative embedding.
              </p>
            )}
          </div>

          {/* Field Mapping — the date/location gates used to look for fixed
              column names (date_time/date, latitude/lat, longitude/lon) and
              silently never fire if a dataset used different ones. These
              selectors point at the real columns; auto-detected defaults
              above keep already-matching datasets working with no clicks. */}
          <div className="mt-3 flex flex-col gap-2 border-t border-border/50 pt-3">
            <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
              <Calendar className="h-3.5 w-3.5" /> Field Mapping (Date & Location)
            </span>
            {data?.columns?.length ? (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <label className="text-[10px] text-text-muted">Date Column</label>
                  <select
                    value={dateCol}
                    onChange={(e) => setDateCol(e.target.value)}
                    className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                  >
                    <option value="">(none — no date gating)</option>
                    {data.columns.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-[10px] text-text-muted">Location Field Type</label>
                  <div className="mt-1 flex gap-1.5">
                    <button
                      onClick={() => setLocationMode('coords')}
                      className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                        locationMode === 'coords' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                      }`}
                    >
                      Lat/Lon
                    </button>
                    <button
                      onClick={() => setLocationMode('name')}
                      className={`flex-1 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                        locationMode === 'name' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                      }`}
                    >
                      Location Name
                    </button>
                  </div>
                </div>

                {locationMode === 'coords' ? (
                  <>
                    <div>
                      <label className="text-[10px] text-text-muted">Latitude Column</label>
                      <select
                        value={latCol}
                        onChange={(e) => setLatCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] text-text-muted">Longitude Column</label>
                      <select
                        value={lonCol}
                        onChange={(e) => setLonCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <label className="text-[10px] text-text-muted">
                        Location Name Column (city) — matched via fuzzy text, or an offline US
                        Census Gazetteer lookup if enabled below
                      </label>
                      <select
                        value={locationCol}
                        onChange={(e) => setLocationCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] text-text-muted">
                        State Column (optional) — improves gazetteer match accuracy; if unset,
                        a combined "City, ST" in the column above is parsed instead
                      </label>
                      <select
                        value={stateCol}
                        onChange={(e) => setStateCol(e.target.value)}
                        className="mt-1 w-full rounded-md border border-border bg-deep px-2.5 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
                      >
                        <option value="">(none)</option>
                        {data.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    <div className="sm:col-span-2 flex items-center justify-between rounded-md border border-border/50 bg-deep/50 px-3 py-2">
                      <span className="text-[10px] text-text-muted">
                        US Census Gazetteer lookup (city/state → real lat/lon centroid) — US-only,
                        and a bare place name can collide across states/countries, so it's opt-in
                        rather than the default; off falls back to fuzzy text match only.
                      </span>
                      <button
                        onClick={() => setUseGazetteer(!useGazetteer)}
                        className={`ml-3 shrink-0 rounded-full border px-3 py-1 text-[11px] font-medium transition-colors ${
                          useGazetteer
                            ? 'border-accent bg-accent-dim/30 text-accent-bright'
                            : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        {useGazetteer ? 'ON' : 'OFF'}
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <p className="text-[11px] text-text-muted">
                Load a dataset to map its date/location columns.
              </p>
            )}
          </div>

          {/* Cluster-Blocked Comparison — the pairwise search below normally
              builds a dense N x N similarity matrix, which stops being
              feasible somewhere in the tens of thousands of rows. Turning
              this on pre-groups rows with UMAP+HDBSCAN (the same technique
              the Analysis tab's cluster pipeline uses) and only compares
              rows within the same group, trading a small recall risk near
              cluster boundaries for comparisons that scale with group size
              instead of dataset size. */}
          <div className="mt-3 flex flex-col gap-2 border-t border-border/50 pt-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-medium text-text-secondary">
                <Layers className="h-3.5 w-3.5" /> Cluster-Blocked Comparison
              </span>
              <button
                onClick={() => setUseClusterBlocking(!useClusterBlocking)}
                className={`rounded-full border px-3 py-1 text-[11px] font-medium transition-colors ${
                  useClusterBlocking
                    ? 'border-accent bg-accent-dim/30 text-accent-bright'
                    : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                }`}
              >
                {useClusterBlocking ? 'ON' : 'OFF'}
              </button>
            </div>
            <p className="text-[10px] text-text-muted">
              Pre-groups rows by embedding similarity (UMAP+HDBSCAN) and only compares rows within the
              same group, instead of every row against every other row. Recommended for datasets past
              roughly 20-30k rows, where the full comparison matrix gets expensive; leave off for smaller
              datasets since a true duplicate pair split across two groups near a boundary would be missed.
            </p>
            {useClusterBlocking && (
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-text-muted">Min group size</label>
                <input
                  type="number"
                  min={2}
                  max={500}
                  value={blockMinClusterSize}
                  onChange={(e) => setBlockMinClusterSize(Math.max(2, parseInt(e.target.value, 10) || 15))}
                  className="w-20 rounded-md border border-border bg-deep px-2 py-1 text-xs text-text-primary focus:border-accent focus:outline-none"
                />
              </div>
            )}
          </div>
        </Panel>
      )}

      {/* ADVANCED TAB */}
      {activeTab === 'advanced' && (
        <div className="flex flex-col gap-4">
          <Panel
            title="Batch Cluster Pipeline"
            subtitle="Stage A screens every candidate pair live; Stage B unions the full-convergence pairs into canonical clusters"
            actions={
              <button
                onClick={handleRunAdvanced}
                disabled={advLoading}
                className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                {advLoading ? 'Running…' : 'Run Pipeline'}
              </button>
            }
          >
            <div className="grid grid-cols-1 items-center gap-6 sm:grid-cols-3">
              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Cosine Similarity Threshold</span>
                  <span className="font-mono text-accent">{threshold}</span>
                </label>
                <input
                  type="range"
                  min="0.50"
                  max="1.00"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Temporal Diff (Days)</span>
                  <span className="font-mono text-accent">{dateDiffDays} d</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="60"
                  step="1"
                  value={dateDiffDays}
                  onChange={(e) => setDateDiffDays(parseInt(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Spatial Radius (Haversine)</span>
                  <span className="font-mono text-accent">{maxKm} km</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="200"
                  step="5"
                  value={maxKm}
                  onChange={(e) => setMaxKm(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>
            </div>

            <DedupeProgressBar loading={advLoading} title="Batch Clustering & Graph Screening" />
          </Panel>

          {/* Results Display */}
          {advResult && advResult.status !== 'success' && (
            <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-2.5 text-sm text-danger">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{advResult.message || 'Advanced deduplication failed.'}</span>
            </div>
          )}
          {advResult && advResult.status === 'success' && (() => {
            const summary = advResult.summary;
            // Full Convergence first, then Date+Location, Text+Location,
            // Text+Date, Text-only, then everything else — similarity
            // descending within each bucket.
            const pairs = [...(advResult.pairs ?? [])].sort((a, b) => {
              const rankDiff = gateSortRank(a) - gateSortRank(b);
              return rankDiff !== 0 ? rankDiff : b.similarity - a.similarity;
            });
            const clusters = advResult.clusters ?? [];
            if (!summary) return null;
            const filteredPairs = pairs.filter((p) => {
              if (auditBinFilter !== 'All' && p.bin !== auditBinFilter) return false;
              if (auditGateFilter !== 'show_all') {
                if (auditGateFilter === 'similar_text' && !p.flags.is_similar_text) return false;
                if (auditGateFilter === 'similar_date' && !p.flags.is_similar_date) return false;
                if (auditGateFilter === 'similar_location' && !p.flags.is_similar_location) return false;
                if (auditGateFilter === 'similar_text_date' && !p.flags.is_similar_text_date) return false;
                if (auditGateFilter === 'similar_both' && !p.flags.is_similar_both) return false;
                if (auditGateFilter === 'similar_all' && !p.flags.is_similar_all) return false;
              }
              return true;
            });
            return (
              <div className="flex flex-col gap-4">
                {/* KPI Cards */}
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
                  <div className="flex flex-col rounded-md border border-border bg-raised p-3">
                    <span className="text-xs text-text-muted">Total Clusters Formed</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-accent">{summary.total_clusters}</span>
                    <span className="mt-1 text-[11px] text-text-muted">Union-find connected components</span>
                  </div>
                  <div className="flex flex-col rounded-md border border-border bg-raised p-3">
                    <span className="text-xs text-text-muted">Rows in Clusters</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-text-primary">{summary.rows_in_clusters}</span>
                    <span className="mt-1 text-[11px] text-text-muted">Total candidate reports linked</span>
                  </div>
                  <div className="flex flex-col rounded-md border border-success/30 bg-success/10 p-3">
                    <span className="text-xs text-success">Redundant Rows Saved</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-success">{summary.redundant_rows_saved}</span>
                    <span className="mt-1 text-[11px] text-success/80">Non-canonical duplicates pruned</span>
                  </div>
                  <div className="flex flex-col rounded-md border border-border bg-raised p-3">
                    <span className="text-xs text-text-muted">Candidate Pairs Screened</span>
                    <span className="mt-1 font-mono text-2xl font-bold text-text-primary">{summary.total_pairs_evaluated}</span>
                    <span className="mt-1 text-[11px] text-text-muted">Stage A multi-gate evaluations</span>
                  </div>
                </div>

                {summary.block_stats && (
                  <p className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <Layers className="h-3 w-3" /> Cluster-blocked: {summary.block_stats.block_count} groups cut the comparison
                    matrix from {summary.block_stats.full_matrix_cells.toLocaleString()} to{' '}
                    {summary.block_stats.compared_cells.toLocaleString()} cells
                    {summary.block_stats.noise_block_size > 0 && ` (${summary.block_stats.noise_block_size} rows ungrouped)`}.
                  </p>
                )}

                {/* Semantic Similarity Bins — cards double as filter toggles */}
                <Panel title="Semantic Similarity Bins" subtitle="Click a bin to filter Stage A below; click again to clear">
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    {([
                      ['exact_duplicate', 'Exact Duplicate (≥0.88)'],
                      ['strong_similar', 'Strong Similar (0.80-0.88)'],
                      ['moderate_similar', 'Moderate Similar (0.70-0.80)'],
                      ['distinct', 'Distinct (<0.70)'],
                    ] as const).map(([key, label]) => (
                      <div
                        key={key}
                        onClick={() => { setAuditBinFilter(auditBinFilter === key ? 'All' : key); setSelectedAuditPairIndex(0); }}
                        className={`flex cursor-pointer flex-col rounded-md border p-3 transition-colors ${
                          auditBinFilter === key
                            ? 'border-accent bg-accent-dim/30'
                            : 'border-border bg-raised hover:border-border-bright'
                        }`}
                      >
                        <span className="text-xs text-text-secondary">{label}</span>
                        <span className="mt-1 font-mono text-2xl font-bold text-text-primary">{summary.bins[key]}</span>
                      </div>
                    ))}
                  </div>
                </Panel>

                {/* Interactive Multi-Gate Filter */}
                <Panel title="Interactive Multi-Gate Filter" subtitle="Select a filter to slice the Stage A pairs below">
                  <div className="flex flex-wrap gap-2">
                    {([
                      ['similar_text', FileText, `Similar Text (${summary.gate_counts.similar_text})`],
                      ['similar_date', Calendar, `Similar Date (${summary.gate_counts.similar_date})`],
                      ['similar_location', MapPin, `Similar Location (${summary.gate_counts.similar_location})`],
                      ['similar_text_date', Puzzle, `Text + Date (${summary.gate_counts.similar_text_date})`],
                      ['similar_both', Zap, `Both (Spatial+Temporal) (${summary.gate_counts.similar_both})`],
                      ['similar_all', Target, `Full Convergence (${summary.gate_counts.similar_all})`],
                      ['show_all', Globe, `Show All (${pairs.length})`],
                    ] as const).map(([key, Icon, label]) => (
                      <button
                        key={key}
                        onClick={() => { setAuditGateFilter(key); setSelectedAuditPairIndex(0); }}
                        className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ${
                          auditGateFilter === key ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        <Icon className="h-3.5 w-3.5" /> {label}
                      </button>
                    ))}
                  </div>
                </Panel>

                {/* Stage A: Pairwise Audit Trail */}
                <Panel
                  title="Stage A — Multi-Gate Pairwise Audit Trail"
                  subtitle="Every candidate pair the screening engine evaluated, with each gate's individual verdict"
                  actions={
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-text-muted">{filteredPairs.length} / {pairs.length} pairs</span>
                      <div className="flex overflow-hidden rounded border border-border">
                        <button
                          onClick={() => setAuditViewMode('table')}
                          className={`px-2.5 py-1 text-[11px] font-medium transition-colors ${
                            auditViewMode === 'table' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                          }`}
                        >
                          Table
                        </button>
                        <button
                          onClick={() => setAuditViewMode('preview')}
                          className={`border-l border-border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                            auditViewMode === 'preview' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                          }`}
                        >
                          Preview
                        </button>
                      </div>
                    </div>
                  }
                  noPad
                >
                  {auditViewMode === 'table' ? (
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse text-left text-xs">
                        <thead>
                          <tr className="border-b border-border bg-deep font-mono text-text-secondary">
                            <th className="p-3">ID A</th>
                            <th className="p-3">ID B</th>
                            <th className="p-3">Similarity</th>
                            <th className="p-3">Distance</th>
                            <th className="p-3">Date Diff</th>
                            <th className="p-3">Witness Notes Preview</th>
                            <th className="p-3">Text</th>
                            <th className="p-3">Date</th>
                            <th className="p-3">Location</th>
                            <th className="p-3">Full Convergence</th>
                            <th className="p-3"></th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40 text-text-primary">
                          {filteredPairs.length === 0 ? (
                            <tr>
                              <td colSpan={11} className="p-6 text-center text-xs text-text-muted">
                                No candidate pairs match the active filters.
                              </td>
                            </tr>
                          ) : filteredPairs.map((pair, idx) => (
                            <tr
                              key={idx}
                              onClick={() => { setSelectedAuditPairIndex(idx); setAuditViewMode('preview'); }}
                              className="cursor-pointer hover:bg-elevated/30 transition-colors"
                            >
                              <td className="p-3 font-mono font-medium text-accent">{pair.id_a}</td>
                              <td className="p-3 font-mono text-text-secondary">{pair.id_b}</td>
                              <td className="p-3 font-mono">
                                <span className="rounded border border-accent/40 bg-accent/10 px-2 py-0.5 text-accent">
                                  {(pair.similarity * 100).toFixed(1)}%
                                </span>
                              </td>
                              <td className="p-3 font-mono text-text-secondary">{pair.haversine_km != null ? `${pair.haversine_km} km` : '—'}</td>
                              <td className="p-3 font-mono text-text-secondary">{pair.date_diff_days != null ? `${pair.date_diff_days} d` : '—'}</td>
                              <td className="max-w-md p-3 text-[11px]">
                                <div className="mb-1 line-clamp-1 border-b border-border/30 pb-1"><span className="font-medium text-text-secondary">A:</span> {pair.text_a_preview}</div>
                                <div className="line-clamp-1"><span className="font-medium text-text-secondary">B:</span> {pair.text_b_preview}</div>
                              </td>
                              {(['is_similar_text', 'is_similar_date', 'is_similar_location', 'is_similar_all'] as const).map((flagKey) => (
                                <td key={flagKey} className="p-3">
                                  <span className={pair.flags[flagKey] ? 'font-medium text-success' : 'text-text-muted'}>
                                    {pair.flags[flagKey] ? '✓' : '—'}
                                  </span>
                                </td>
                              ))}
                              <td className="p-3">
                                <button
                                  onClick={(e) => { e.stopPropagation(); setSelectedAuditPairIndex(idx); setAuditViewMode('preview'); }}
                                  className="flex items-center gap-1 rounded border border-border px-2 py-1 text-[11px] text-text-muted transition-colors hover:border-accent hover:text-accent"
                                >
                                  <Layers className="h-3 w-3" /> Preview
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : filteredPairs[selectedAuditPairIndex] || filteredPairs[0] ? (
                    <AlignedInspector pair={filteredPairs[selectedAuditPairIndex] || filteredPairs[0]} />
                  ) : (
                    <div className="p-12 text-center text-sm text-text-muted">
                      No candidate pairs to preview — adjust the filters above.
                    </div>
                  )}
                </Panel>

                {/* Stage B: Canonical Clusters */}
                <Panel
                  title="Stage B — Canonical Clusters"
                  subtitle="Connected components over the full-convergence pairs above, one canonical (most-complete) row picked per cluster"
                  actions={
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs text-text-muted">{clusters.length} clusters</span>
                      <button
                        onClick={handleApplyDedup}
                        disabled={applyLoading}
                        title="Loads this dataset (with dedup_cluster_id/size/is_canonical/is_duplicate columns baked in) as the active Data Explorer dataset — no download/re-upload needed"
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                      >
                        <Upload className="h-3 w-3" /> {applyLoading ? 'Applying...' : 'Apply to Dataset'}
                      </button>
                      <button
                        onClick={handleExportDedup}
                        disabled={exportLoading}
                        title="Downloads the full dataset with dedup_cluster_id/size/is_canonical/is_duplicate columns baked in, plus a metadata JSON recording this run's parameters — zipped together"
                        className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-[11px] text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:opacity-50"
                      >
                        <Download className="h-3 w-3" /> {exportLoading ? 'Exporting...' : 'Export Enriched Dataset (.zip)'}
                      </button>
                    </div>
                  }
                  noPad
                >
                  {clusters.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse text-left text-xs">
                        <thead>
                          <tr className="border-b border-border bg-deep font-mono text-text-secondary">
                            <th className="p-3">Cluster ID</th>
                            <th className="p-3">Size</th>
                            <th className="p-3">Canonical Row ID</th>
                            <th className="p-3">Member IDs</th>
                            <th className="p-3"></th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40 text-text-primary">
                          {clusters.map((c) => (
                            <tr key={c.cluster_id} className="hover:bg-elevated/30 transition-colors">
                              <td className="p-3 font-mono font-medium text-accent">{c.cluster_id}</td>
                              <td className="p-3 font-mono">{c.size}</td>
                              <td className="p-3 font-mono text-success">{c.canonical_id}</td>
                              <td className="p-3 font-mono text-[11px] text-text-secondary">{c.member_ids.join(', ')}</td>
                              <td className="p-3">
                                <button
                                  onClick={() => setSelectedClusterId(selectedClusterId === c.cluster_id ? null : c.cluster_id)}
                                  className="rounded border border-border px-2 py-1 text-[11px] text-text-muted transition-colors hover:border-accent hover:text-accent"
                                >
                                  {selectedClusterId === c.cluster_id ? 'Hide' : 'Preview'}
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="p-4 text-xs text-text-muted">
                      No clusters formed — no pairs reached full convergence (text + date + location) at the current settings.
                    </p>
                  )}

                  {selectedClusterId && (() => {
                    const c = clusters.find((cl) => cl.cluster_id === selectedClusterId);
                    if (!c) return null;
                    // Fall back to whatever keys actually showed up if the run
                    // had no explicit column selection (preview_cols empty).
                    const cols = c.preview_cols.length > 0
                      ? c.preview_cols
                      : Array.from(new Set(c.member_previews.flatMap((m) => Object.keys(m.values))));
                    return (
                      <div className="border-t border-border bg-deep p-4">
                        <span className="text-xs font-medium text-text-secondary">
                          {c.cluster_id} — {c.member_previews.length} members, selected columns
                        </span>
                        <div className="mt-2 overflow-x-auto rounded-md border border-border">
                          <table className="w-full border-collapse text-left text-[11px]">
                            <thead>
                              <tr className="border-b border-border bg-abyss font-mono text-text-secondary">
                                <th className="p-2">Column</th>
                                {c.member_previews.map((m) => (
                                  <th
                                    key={m.id}
                                    className={`border-l border-border/50 p-2 ${m.id === c.canonical_id ? 'text-accent' : 'text-text-primary'}`}
                                  >
                                    {m.id}{m.id === c.canonical_id ? ' (canonical)' : ''}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-border/40 text-text-primary">
                              {cols.map((col) => (
                                <tr key={col} className="hover:bg-elevated/30 transition-colors">
                                  <td className="bg-abyss/60 p-2 font-mono font-medium text-text-secondary">{col}</td>
                                  {c.member_previews.map((m) => (
                                    <td key={m.id} className="border-l border-border/50 p-2 align-top">
                                      {m.values[col] == null || m.values[col] === '' ? '—' : String(m.values[col])}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    );
                  })()}
                </Panel>
              </div>
            );
          })()}
        </div>
      )}

      {/* CROSS-DB TAB */}
      {activeTab === 'cross_db' && (
        <div className="flex flex-col gap-4">
          <Panel
            title="Cross-DB Pipeline"
            subtitle="Compare records within a single dataset or across two databases using Harrier cosine similarity"
            actions={
              <button
                onClick={handleRunCrossDb}
                disabled={crossLoading}
                className="flex items-center gap-2 rounded-md bg-accent-dim px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                {crossLoading ? 'Running…' : 'Execute Pipeline'}
              </button>
            }
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium text-text-secondary">Comparison Scope:</span>
              <button
                onClick={() => setCrossCmpMode('within')}
                className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                  crossCmpMode === 'within' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                }`}
              >
                Within Dataset (DB1 vs DB1)
              </button>
              <button
                onClick={() => setCrossCmpMode('between')}
                className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                  crossCmpMode === 'between' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                }`}
              >
                Between Datasets (DB1 vs DB2)
              </button>
            </div>

            <div className="mt-3 grid grid-cols-1 items-center gap-6 border-t border-border/50 pt-3 sm:grid-cols-3">
              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Cosine Similarity Threshold</span>
                  <span className="font-mono text-accent">{threshold}</span>
                </label>
                <input
                  type="range"
                  min="0.50"
                  max="1.00"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Temporal Diff (Days)</span>
                  <span className="font-mono text-accent">{dateDiffDays} d</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="60"
                  step="1"
                  value={dateDiffDays}
                  onChange={(e) => setDateDiffDays(parseInt(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>

              <div>
                <label className="flex justify-between text-xs text-text-secondary">
                  <span>Max Spatial Radius (Haversine)</span>
                  <span className="font-mono text-accent">{maxKm} km</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="200"
                  step="5"
                  value={maxKm}
                  onChange={(e) => setMaxKm(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>
            </div>

            <DedupeProgressBar loading={crossLoading} title="Cross-Database Similarity Pipeline" />
          </Panel>

          {crossResult && (() => {
            // Bins and gates are both always-on filters over the same pairs
            // list now (no more separate Easy/Hard "path" — apply both).
            const filteredPairs = crossResult.pairs.filter((p) => {
              if (crossBinFilter !== 'All' && p.bin !== crossBinFilter) return false;
              if (crossGateFilter !== 'show_all') {
                if (crossGateFilter === 'similar_text' && !p.flags.is_similar_text) return false;
                if (crossGateFilter === 'similar_date' && !p.flags.is_similar_date) return false;
                if (crossGateFilter === 'similar_location' && !p.flags.is_similar_location) return false;
                if (crossGateFilter === 'similar_text_date' && !p.flags.is_similar_text_date) return false;
                if (crossGateFilter === 'similar_both' && !p.flags.is_similar_both) return false;
                if (crossGateFilter === 'similar_all' && !p.flags.is_similar_all) return false;
              }
              return true;
            });
            const activePair = filteredPairs[selectedPairIndex] || filteredPairs[0] || crossResult.pairs[0];
            const openPreview = (idx: number) => { setSelectedPairIndex(idx); setCrossViewMode('preview'); };

            return (
              <div className="flex flex-col gap-4">
                {crossResult.summary.block_stats && (
                  <p className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <Layers className="h-3 w-3" /> Cluster-blocked: {crossResult.summary.block_stats.block_count} groups cut the
                    comparison matrix from {crossResult.summary.block_stats.full_matrix_cells.toLocaleString()} to{' '}
                    {crossResult.summary.block_stats.compared_cells.toLocaleString()} cells
                    {crossResult.summary.block_stats.noise_block_size > 0 &&
                      ` (${crossResult.summary.block_stats.noise_block_size} rows ungrouped)`}.
                  </p>
                )}

                {/* Semantic Similarity Bins — cards double as filter toggles */}
                <Panel title="Semantic Similarity Bins" subtitle="Click a bin to filter the table below; click again to clear">
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    {([
                      ['exact_duplicate', 'Exact Duplicate (≥0.88)'],
                      ['strong_similar', 'Strong Similar (0.80-0.88)'],
                      ['moderate_similar', 'Moderate Similar (0.70-0.80)'],
                      ['distinct', 'Distinct (<0.70)'],
                    ] as const).map(([key, label]) => (
                      <div
                        key={key}
                        onClick={() => setCrossBinFilter(crossBinFilter === key ? 'All' : key)}
                        className={`flex cursor-pointer flex-col rounded-md border p-3 transition-colors ${
                          crossBinFilter === key
                            ? 'border-accent bg-accent-dim/30'
                            : 'border-border bg-raised hover:border-border-bright'
                        }`}
                      >
                        <span className="text-xs text-text-secondary">{label}</span>
                        <span className="mt-1 font-mono text-2xl font-bold text-text-primary">{crossResult.summary.bins[key]}</span>
                      </div>
                    ))}
                  </div>
                </Panel>

                {/* Interactive Multi-Gate Filter */}
                <Panel title="Interactive Multi-Gate Filter" subtitle="Select a filter to slice the candidate pairs below">
                  <div className="flex flex-wrap gap-2">
                    {([
                      ['similar_text', FileText, `Similar Text (${crossResult.summary.gate_counts.similar_text})`],
                      ['similar_date', Calendar, `Similar Date (${crossResult.summary.gate_counts.similar_date})`],
                      ['similar_location', MapPin, `Similar Location (${crossResult.summary.gate_counts.similar_location})`],
                      ['similar_text_date', Puzzle, `Text + Date (${crossResult.summary.gate_counts.similar_text_date})`],
                      ['similar_both', Zap, `Both (Spatial+Temporal) (${crossResult.summary.gate_counts.similar_both})`],
                      ['similar_all', Target, `Full Convergence (${crossResult.summary.gate_counts.similar_all})`],
                      ['show_all', Globe, `Show All (${crossResult.pairs.length})`],
                    ] as const).map(([key, Icon, label]) => (
                      <button
                        key={key}
                        onClick={() => { setCrossGateFilter(key); setSelectedPairIndex(0); }}
                        className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors ${
                          crossGateFilter === key ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        <Icon className="h-3.5 w-3.5" /> {label}
                      </button>
                    ))}
                  </div>
                </Panel>

                {/* Unified Candidate Pairs — full-width table, mixing the old
                    "Candidate Pair Evaluation Matrix" and "Batch Aligned
                    Table" columns, plus a Preview tab that swaps in the
                    Aligned Inspector for whichever pair was clicked. */}
                <Panel
                  title="Candidate Pairs"
                  actions={
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setCrossViewMode('table')}
                        className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                          crossViewMode === 'table' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        Table ({filteredPairs.length})
                      </button>
                      <button
                        onClick={() => setCrossViewMode('preview')}
                        className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
                          crossViewMode === 'preview' ? 'border-accent bg-accent-dim/30 text-accent-bright' : 'border-border bg-raised text-text-secondary hover:border-border-bright'
                        }`}
                      >
                        Preview
                      </button>
                    </div>
                  }
                  noPad
                >
                  {crossViewMode === 'table' ? (
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse text-left text-xs">
                        <thead>
                          <tr className="border-b border-border bg-deep font-mono text-text-secondary">
                            <th className="p-3">Pair ID (A ↔ B)</th>
                            <th className="p-3">Score</th>
                            <th className="p-3">Bin</th>
                            <th className="p-3">Dates (A vs B)</th>
                            <th className="p-3">Location (A vs B)</th>
                            <th className="p-3">Witness Notes Preview</th>
                            <th className="p-3">Multi-Gate Status</th>
                            <th className="p-3"></th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40 text-text-primary">
                          {filteredPairs.length === 0 ? (
                            <tr>
                              <td colSpan={8} className="p-6 text-center text-xs text-text-muted">
                                No candidate pairs match the active filters.
                              </td>
                            </tr>
                          ) : (
                            filteredPairs.map((pair, idx) => (
                              <tr
                                key={idx}
                                onClick={() => openPreview(idx)}
                                className="cursor-pointer hover:bg-elevated/30 transition-colors"
                              >
                                <td className="whitespace-nowrap p-3 font-mono font-medium text-accent">
                                  {pair.id_a} <span className="text-text-muted">↔</span> {pair.id_b}
                                </td>
                                <td className="p-3 font-mono font-medium">{(pair.similarity * 100).toFixed(1)}%</td>
                                <td className="p-3 font-mono">
                                  <span className="rounded border border-border bg-elevated px-2 py-0.5 text-xs">{pair.bin}</span>
                                </td>
                                <td className="whitespace-nowrap p-3 font-mono text-[11px]">
                                  <div><span className="text-text-muted">A:</span> {String(pair.row_a?.date_time || pair.row_a?.date || '—')}</div>
                                  <div><span className="text-text-muted">B:</span> {String(pair.row_b?.date_time || pair.row_b?.date || '—')}</div>
                                  <div className="mt-0.5 text-[10px] text-text-muted">Gap: {pair.date_diff_days ?? '—'}d</div>
                                </td>
                                <td className="whitespace-nowrap p-3 font-mono text-[11px]">
                                  {pair.haversine_km !== null ? (
                                    <>
                                      <div><span className="text-text-muted">A:</span> {pair.row_a?.latitude !== undefined ? `${Number(pair.row_a.latitude).toFixed(2)}, ${Number(pair.row_a.longitude).toFixed(2)}` : '—'}</div>
                                      <div><span className="text-text-muted">B:</span> {pair.row_b?.latitude !== undefined ? `${Number(pair.row_b.latitude).toFixed(2)}, ${Number(pair.row_b.longitude).toFixed(2)}` : '—'}</div>
                                      <div className="mt-0.5 text-[10px] text-text-muted">Dist: {pair.haversine_km}km</div>
                                    </>
                                  ) : pair.location_name_similarity !== null ? (
                                    <div className="text-[10px] text-text-muted">Name sim: {(pair.location_name_similarity * 100).toFixed(0)}%</div>
                                  ) : (
                                    <span className="text-text-muted">—</span>
                                  )}
                                </td>
                                <td className="max-w-md p-3 text-[11px]">
                                  <div className="mb-1 line-clamp-1 border-b border-border/30 pb-1"><span className="font-medium text-text-secondary">A:</span> {pair.text_a_preview}</div>
                                  <div className="line-clamp-1"><span className="font-medium text-text-secondary">B:</span> {pair.text_b_preview}</div>
                                </td>
                                <td className="p-3">
                                  <div className="flex gap-1">
                                    <span title="Similar Text" className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${pair.flags.is_similar_text ? 'border border-success/30 bg-success/10 text-success' : 'text-text-muted opacity-40'}`}>TXT</span>
                                    <span title="Similar Date" className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${pair.flags.is_similar_date ? 'border border-success/30 bg-success/10 text-success' : 'text-text-muted opacity-40'}`}>DAT</span>
                                    <span title="Similar Location" className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${pair.flags.is_similar_location ? 'border border-success/30 bg-success/10 text-success' : 'text-text-muted opacity-40'}`}>LOC</span>
                                    <span title="Full Duplicate" className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${pair.flags.is_similar_all ? 'border border-success/30 bg-success/10 text-success' : 'text-text-muted opacity-40'}`}>ALL</span>
                                  </div>
                                </td>
                                <td className="p-3">
                                  <button
                                    onClick={(e) => { e.stopPropagation(); openPreview(idx); }}
                                    className="flex items-center gap-1 rounded border border-border px-2 py-1 text-[11px] text-text-muted transition-colors hover:border-accent hover:text-accent"
                                  >
                                    <Layers className="h-3 w-3" /> Preview
                                  </button>
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  ) : activePair ? (
                    <AlignedInspector pair={activePair} />
                  ) : (
                    <div className="p-12 text-center text-sm text-text-muted">
                      No candidate pairs to preview — adjust the filters above.
                    </div>
                  )}
                </Panel>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}

