import { useState } from 'react';
import { useStore } from '../../store/useStore';
import { Compass, Loader2, X } from 'lucide-react';

interface MagneticGraph {
    title: string;
    station: string;
    iaga: string;
    distance_km: number;
    event_date: string;
    image: string;
}

interface MagneticResult {
    status: string;
    candidates: number;
    scanned: number;
    matched: number;
    skipped_no_station: number;
    skipped_no_data: number;
    distance_km: number;
    elapsed_s: number;
    graphs: MagneticGraph[];
    aggregate: { title: string; image: string } | null;
    message: string;
}

export function MagneticPage() {
    const { dataLoaded, summary } = useStore();

    const columns = summary?.columns || [];

    const findBestMatch = (keywords: string[]) => {
        return columns.find(c => keywords.some(k => c.toLowerCase().includes(k))) || columns[0];
    };

    const [latCol, setLatCol] = useState(findBestMatch(['lat', 'latitude']) || '');
    const [lonCol, setLonCol] = useState(findBestMatch(['lon', 'lng', 'longitude']) || '');
    const [dateCol, setDateCol] = useState(findBestMatch(['date', 'time', 'datetime']) || '');
    const [distance, setDistance] = useState(100);

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<MagneticResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [lightbox, setLightbox] = useState<string | null>(null);

    if (!dataLoaded || !summary) {
        return (
            <div className="flex h-full flex-col items-center justify-center space-y-4">
                <h2 className="text-xl font-bold text-text-primary">No Data Loaded</h2>
                <p className="text-text-secondary">
                    Please load or upload a dataset in the Data Explorer first.
                </p>
            </div>
        );
    }

    const runAnalysis = async () => {
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const res = await fetch('/api/magnetic/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lat_col: latCol,
                    lon_col: lonCol,
                    date_col: dateCol,
                    distance: distance
                })
            });
            if (!res.ok) {
                const body = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(body.detail || `Request failed: ${res.status}`);
            }
            const data: MagneticResult = await res.json();
            setResult(data);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'An error occurred while running magnetic analysis.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex h-full flex-col p-6 space-y-6 max-h-[100vh] overflow-y-auto">
            <div>
                <h1 className="flex items-center gap-2 text-2xl font-bold text-text-primary tracking-tight">
                    <Compass className="h-6 w-6 text-accent" />
                    Magnetic Analysis
                </h1>
                <p className="text-text-secondary mt-1">Cross-reference UAP events with geomagnetic observatory data.</p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <h2 className="text-lg font-semibold text-text-primary mb-4">Configuration</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-text-secondary mb-1">Latitude Column</label>
                            <select
                                value={latCol}
                                onChange={e => setLatCol(e.target.value)}
                                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-primary outline-none focus:border-accent transition-colors"
                            >
                                {columns.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-text-secondary mb-1">Longitude Column</label>
                            <select
                                value={lonCol}
                                onChange={e => setLonCol(e.target.value)}
                                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-primary outline-none focus:border-accent transition-colors"
                            >
                                {columns.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-text-secondary mb-1">Date Column</label>
                            <select
                                value={dateCol}
                                onChange={e => setDateCol(e.target.value)}
                                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-primary outline-none focus:border-accent transition-colors"
                            >
                                {columns.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-text-secondary mb-1">Observatory Search Distance (km)</label>
                            <input
                                type="number"
                                value={distance}
                                onChange={e => setDistance(Number(e.target.value))}
                                min="0"
                                className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-primary outline-none focus:border-accent transition-colors"
                            />
                            <p className="mt-1 text-xs text-text-muted">
                                Higher values match more events but to a more distant observatory. Up to 25 events are scanned per run.
                            </p>
                        </div>
                        <button
                            onClick={runAnalysis}
                            disabled={loading}
                            className="w-full flex justify-center items-center gap-2 rounded-md bg-accent px-4 py-2 font-medium text-white hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors mt-4"
                        >
                            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                            Run Magnetic Analysis
                        </button>
                    </div>
                </div>

                <div className="rounded-xl border border-border bg-card p-6 shadow-sm flex flex-col">
                    <h2 className="text-lg font-semibold text-text-primary mb-4">Run Status</h2>

                    <div className="flex-1 rounded-md border border-dashed border-border p-4 flex flex-col items-center justify-center text-center">
                        {!loading && !result && !error && (
                            <p className="text-text-muted text-sm">Configure parameters and run analysis to cross-reference with BGS observatories.</p>
                        )}

                        {loading && (
                            <div className="flex flex-col items-center gap-2">
                                <Loader2 className="h-8 w-8 animate-spin text-accent" />
                                <p className="text-sm font-medium text-text-secondary">Querying BGS Observatories...</p>
                                <p className="text-xs text-text-muted mt-2 max-w-[250px]">
                                    Fetching X/Y/Z/S minute data per event and running FastDTW alignment. This can take 1-2 minutes.
                                </p>
                            </div>
                        )}

                        {error && (
                            <div className="text-red-500 bg-red-500/10 p-4 rounded-md text-sm border border-red-500/20 w-full text-left">
                                <strong>Error:</strong> {error}
                            </div>
                        )}

                        {result && (
                            <div className="text-left w-full space-y-3">
                                <div className="bg-green-500/10 text-green-500 p-3 rounded-md border border-green-500/20 text-sm">
                                    <span className="font-bold">Done.</span> {result.message}
                                </div>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <Stat label="Candidate events" value={result.candidates} />
                                    <Stat label="Events scanned" value={result.scanned} />
                                    <Stat label="Graphs produced" value={result.matched} />
                                    <Stat label="No observatory in range" value={result.skipped_no_station} />
                                    <Stat label="No usable data" value={result.skipped_no_data} />
                                    <Stat label="Elapsed" value={`${result.elapsed_s}s`} />
                                </div>
                                {result.matched === 0 && (
                                    <p className="text-xs text-text-muted">
                                        No graphs produced — try increasing the search distance so events can match an observatory.
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {result && result.aggregate && (
                <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <h2 className="text-lg font-semibold text-text-primary mb-1">Aggregate</h2>
                    <p className="text-sm text-text-secondary mb-4">{result.aggregate.title}</p>
                    <img
                        src={result.aggregate.image}
                        alt={result.aggregate.title}
                        onClick={() => setLightbox(result.aggregate!.image)}
                        className="mx-auto max-h-[640px] cursor-zoom-in rounded-md border border-border"
                    />
                </div>
            )}

            {result && result.graphs.length > 0 && (
                <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
                    <h2 className="text-lg font-semibold text-text-primary mb-1">
                        Event Graphs <span className="text-text-muted">({result.graphs.length})</span>
                    </h2>
                    <p className="text-sm text-text-secondary mb-4">
                        Geomagnetic X/Y/Z/S variation around each UAP event. Click a graph to enlarge.
                    </p>
                    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                        {result.graphs.map((g, i) => (
                            <div key={i} className="rounded-lg border border-border bg-surface overflow-hidden">
                                <img
                                    src={g.image}
                                    alt={g.title}
                                    onClick={() => setLightbox(g.image)}
                                    className="w-full cursor-zoom-in bg-[#0d0d0d]"
                                />
                                <div className="p-3">
                                    <p className="text-sm font-medium text-text-primary truncate" title={g.title}>
                                        {g.station}
                                    </p>
                                    <p className="text-xs text-text-muted">
                                        {g.event_date.slice(0, 10)} · {g.iaga} · {g.distance_km} km away
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {lightbox && (
                <div
                    onClick={() => setLightbox(null)}
                    className="fixed inset-0 z-[120] flex items-center justify-center bg-background/90 p-6 backdrop-blur-sm"
                >
                    <button
                        onClick={() => setLightbox(null)}
                        className="absolute right-6 top-6 rounded-full bg-elevated p-2 text-text-secondary hover:text-text-primary"
                    >
                        <X className="h-5 w-5" />
                    </button>
                    <img
                        src={lightbox}
                        alt="Magnetic graph"
                        onClick={e => e.stopPropagation()}
                        className="max-h-[90vh] max-w-[90vw] rounded-md border border-border"
                    />
                </div>
            )}
        </div>
    );
}

function Stat({ label, value }: { label: string; value: string | number }) {
    return (
        <div className="rounded-md border border-border/60 bg-surface px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-text-muted">{label}</p>
            <p className="text-base font-semibold text-text-primary">{value}</p>
        </div>
    );
}
