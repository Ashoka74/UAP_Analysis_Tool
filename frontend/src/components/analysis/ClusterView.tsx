import { Panel } from '../common/Panel';

export function ClusterView() {
    return (
        <div className="h-full space-y-4">
            <Panel
                title="EDA Clusters (LLM Analysis)"
                subtitle="Advanced clustering of UAP sightings using SentenceTransformers and HDBSCAN"
                noPad
            >
                <div className="relative h-[calc(100vh-180px)] w-full overflow-hidden rounded-b-lg">
                    <iframe
                        src="/api/analysis/clusters"
                        className="h-full w-full border-0"
                        title="UAP Clusters LLM"
                    />
                </div>
            </Panel>
        </div>
    );
}
