import { Panel } from '../common/Panel';
import { BASE } from '../../api/client';

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
                        src={`${BASE}/analysis/clusters`}
                        className="h-full w-full border-0"
                        title="UAP Clusters LLM"
                    />

                </div>
            </Panel>
        </div>
    );
}
