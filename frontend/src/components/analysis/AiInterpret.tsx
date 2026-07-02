import { useState } from 'react';
import { Sparkles, AlertTriangle, Key, X, Loader2 } from 'lucide-react';
import { api } from '../../api/client';
import { useStore } from '../../store/useStore';
import { Markdown } from '../common/Markdown';

interface Props {
  kind: 'cramers' | 'xgboost';
  // Builds the rendered table/results text handed to the AI (called on click).
  buildContext: () => string;
  label?: string;
}

/**
 * "Interpret with AI" — a sparkle button that posts the current results table to
 * Gemini and renders a plain-language reading. Mirrors the AI Query page (key
 * from the store, never stored server-side); the per-kind prompt lives in the
 * backend so the caveats stay consistent.
 */
export function AiInterpret({ kind, buildContext, label = 'Interpret with AI' }: Props) {
  const { geminiKey, setGeminiKey } = useStore();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setOpen(true);
    setError(null);
    if (!geminiKey) {
      setError('Enter your Gemini API key, then click Interpret again.');
      return;
    }
    const context = buildContext();
    if (!context.trim()) {
      setError('Nothing to interpret yet — compute results first.');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await api.interpretAnalysis(kind, context, geminiKey);
      setResult(res.response);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'AI interpretation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      <button
        onClick={run}
        disabled={loading}
        title="Send this table to Gemini for a plain-language interpretation"
        className="flex items-center gap-1.5 rounded-md border border-purple/40 bg-purple/10 px-3 py-1.5 text-xs font-medium text-purple transition-colors hover:bg-purple/20 disabled:opacity-50"
      >
        <Sparkles className="h-3.5 w-3.5" />
        {loading ? 'Interpreting…' : label}
      </button>

      {open && (
        <div className="rounded-md border border-purple/30 bg-surface">
          <div className="flex items-center justify-between border-b border-border px-3 py-2">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-text-secondary">
              <Sparkles className="h-3.5 w-3.5 text-purple" /> AI interpretation
              <span className="font-normal text-text-muted">· Gemini</span>
            </span>
            <button onClick={() => setOpen(false)} className="text-text-muted transition-colors hover:text-text-primary">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="space-y-2 p-3 text-xs">
            {!geminiKey && (
              <div>
                <label className="mb-1 block text-text-muted">Gemini API key</label>
                <div className="relative">
                  <Key className="absolute left-2.5 top-2 h-3.5 w-3.5 text-text-muted" />
                  <input
                    type="password"
                    value={geminiKey}
                    onChange={(e) => setGeminiKey(e.target.value)}
                    placeholder="Paste key, then click Interpret again…"
                    className="w-full rounded border border-border bg-deep py-1.5 pl-8 pr-3 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
                  />
                </div>
              </div>
            )}
            {error && (
              <div className="flex items-center gap-2 rounded bg-danger/10 px-2.5 py-1.5 text-danger">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> {error}
              </div>
            )}
            {loading && (
              <div className="flex items-center gap-2 text-text-muted">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Analyzing the results…
              </div>
            )}
            {result && <Markdown>{result}</Markdown>}
          </div>
        </div>
      )}
    </div>
  );
}
