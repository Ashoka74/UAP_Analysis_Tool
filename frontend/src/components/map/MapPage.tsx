import { useState } from 'react';
import { useStore } from '../../store/useStore';
import { Loader2 } from 'lucide-react';

export function MapPage() {
    const { dataLoaded, summary } = useStore();
    const [loading, setLoading] = useState(true);

    const [refreshKey, setRefreshKey] = useState(0);

    const handleRefresh = () => {
        setLoading(true);
        setRefreshKey(prev => prev + 1);
    };

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

    return (
        <div className="flex h-full flex-col p-6 space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-text-primary tracking-tight">Geospatial Analysis</h1>
                    <p className="text-text-secondary mt-1">Interactive Kepler.gl Map</p>
                </div>
                <button
                    onClick={handleRefresh}
                    className="px-4 py-2 bg-accent/10 hover:bg-accent/20 text-accent rounded-lg border border-accent/20 transition-colors flex items-center gap-2"
                >
                    <Loader2 className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                    Refresh Map
                </button>
            </div>

            <div className="relative flex-1 rounded-xl border border-border bg-card overflow-hidden shadow-lg shadow-black/20">
                {loading && (
                    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-card/80 backdrop-blur-sm">
                        <Loader2 className="h-8 w-8 animate-spin text-accent mb-4" />
                        <p className="text-text-secondary font-medium">Generating interactive map...</p>
                    </div>
                )}
                <iframe
                    key={refreshKey}
                    src="/api/map/html"
                    className="w-full h-full border-0 absolute inset-0 z-0"
                    onLoad={() => setLoading(false)}
                    title="Kepler Map"
                    allowFullScreen
                />
            </div>
        </div>
    );
}
