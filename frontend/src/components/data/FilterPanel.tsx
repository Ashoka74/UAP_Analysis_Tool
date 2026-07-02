import { useState, useEffect, useRef } from 'react';
import { Filter, X, Plus } from 'lucide-react';
import { api } from '../../api/client';
import type { ColumnStat } from '../../types';

interface ActiveFilter {
  id: number;
  column: string;
  type: string;
  values?: string[];
  min_val?: number;
  max_val?: number;
  pattern?: string;
  start?: string;
  end?: string;
  drop_null?: boolean;
}

// Filter types the user can pick per column (mirrors the Streamlit DataProcessor:
// categorical multiselect, numeric range, date range, text regex). Binary columns
// are filtered as categorical.
const TYPE_OPTIONS = [
  { id: 'categorical', label: 'Categorical' },
  { id: 'numeric', label: 'Numeric' },
  { id: 'date', label: 'Date' },
  { id: 'text', label: 'Text' },
];

// Best-guess default type for a column (the user can override via the selector).
function defaultType(c?: ColumnStat): string {
  if (!c) return 'text';
  const dt = (c.dtype || '').toLowerCase();
  if (dt.includes('date') || dt.includes('time')) return 'date';
  if (c.top_values) return 'categorical';
  if (c.min != null) return 'numeric';
  return 'text';
}

interface FilterPanelProps {
  columns: ColumnStat[];
  onApply: (filters: ActiveFilter[]) => void;
}

type ValueCount = { value: string; count: number };

let filterId = 0;

/**
 * Searchable value picker for categorical filters.
 *
 * Shows the column's precomputed top values (from client-side column stats)
 * immediately, so values are always visible the moment the field is clicked.
 * Typing queries the server for the full set of matching values; if the
 * server is unavailable the picker falls back to filtering the local list.
 */
function CategoricalValuePicker({
  column,
  selected,
  onChange,
  fallbackValues,
}: {
  column: string;
  selected: string[];
  onChange: (values: string[]) => void;
  fallbackValues: ValueCount[];
}) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ValueCount[]>(fallbackValues);
  const [totalMatches, setTotalMatches] = useState(fallbackValues.length);
  const [loading, setLoading] = useState(false);
  const [serverBacked, setServerBacked] = useState(true);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const localFilter = (q: string): ValueCount[] => {
    const t = q.trim().toLowerCase();
    return t ? fallbackValues.filter((v) => v.value.toLowerCase().includes(t)) : fallbackValues;
  };

  // Reset to the new column's local values when the column changes.
  useEffect(() => {
    setQuery('');
    setResults(fallbackValues);
    setTotalMatches(fallbackValues.length);
    setServerBacked(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [column]);

  // Debounced server search; falls back to the local list on failure.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const t = setTimeout(() => {
      api
        .getColumnValues(column, query, 50)
        .then((res) => {
          if (cancelled) return;
          setServerBacked(true);
          setResults(res.values);
          setTotalMatches(res.total_matches);
        })
        .catch(() => {
          if (cancelled) return;
          // Server (or its session data) unavailable — use the local list.
          setServerBacked(false);
          const local = localFilter(query);
          setResults(local);
          setTotalMatches(local.length);
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [column, query]);

  // Close the dropdown when clicking outside.
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const toggle = (value: string) => {
    if (selected.includes(value)) onChange(selected.filter((v) => v !== value));
    else onChange([...selected, value]);
  };

  return (
    <div ref={ref} className="relative min-w-0 flex-1">
      {selected.length > 0 && (
        <div className="mb-1 flex flex-wrap gap-1">
          {selected.map((v) => (
            <span
              key={v}
              className="flex items-center gap-1 rounded bg-accent-dim px-1.5 py-0.5 text-[10px] text-white"
            >
              <span className="max-w-[140px] truncate" title={v}>
                {v}
              </span>
              <button onClick={() => toggle(v)} className="hover:text-danger">
                <X className="h-2.5 w-2.5" />
              </button>
            </span>
          ))}
        </div>
      )}

      <input
        type="text"
        placeholder={selected.length ? 'Add another value...' : 'Search values...'}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => setOpen(true)}
        className="w-full rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary placeholder:text-text-muted"
      />

      {open && (
        <div className="absolute left-0 right-0 top-full z-20 mt-1 max-h-48 overflow-auto rounded border border-border bg-deep shadow-xl">
          {results.length === 0 && loading && (
            <div className="px-2 py-1.5 text-[10px] text-text-muted">Searching...</div>
          )}
          {results.length === 0 && !loading && (
            <div className="px-2 py-1.5 text-[10px] text-text-muted">No matching values</div>
          )}
          {results.map((r) => {
            const isSel = selected.includes(r.value);
            return (
              <button
                key={r.value}
                onClick={() => toggle(r.value)}
                className={`flex w-full items-center justify-between gap-2 px-2 py-1 text-left text-[11px] hover:bg-elevated ${
                  isSel ? 'text-accent' : 'text-text-secondary'
                }`}
              >
                <span className="truncate" title={r.value}>
                  {isSel ? '✓ ' : ''}
                  {r.value}
                </span>
                <span className="shrink-0 text-text-muted">{r.count}</span>
              </button>
            );
          })}
          {totalMatches > results.length && (
            <div className="px-2 py-1.5 text-[10px] text-text-muted">
              +{totalMatches - results.length} more — refine your search
            </div>
          )}
          {!serverBacked && (
            <div className="border-t border-border px-2 py-1.5 text-[10px] text-amber-400/80">
              Showing cached top values — reload the dataset for full search.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function FilterPanel({ columns, onApply }: FilterPanelProps) {
  const [filters, setFilters] = useState<ActiveFilter[]>([]);
  const [open, setOpen] = useState(false);

  const addFilter = () => {
    if (columns.length === 0) return;
    const col = columns[0];
    setFilters([...filters, { id: ++filterId, column: col.name, type: defaultType(col) }]);
  };

  const removeFilter = (id: number) => {
    const next = filters.filter((f) => f.id !== id);
    setFilters(next);
    onApply(next);
  };

  const updateFilter = (id: number, patch: Partial<ActiveFilter>) => {
    setFilters((prev) => prev.map((f) => (f.id === id ? { ...f, ...patch } : f)));
  };

  const handleApply = () => {
    onApply(filters);
  };

  return (
    <div className="rounded-lg border border-border bg-surface">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-sm text-text-secondary hover:text-text-primary"
      >
        <Filter className="h-4 w-4" />
        Filters {filters.length > 0 && `(${filters.length})`}
      </button>

      {open && (
        <div className="border-t border-border px-4 py-3">
          <div className="space-y-2">
            {filters.map((f) => {
              const colInfo = columns.find((c) => c.name === f.column);
              return (
                <div key={f.id} className="rounded bg-raised p-2">
                <div className="flex items-start gap-2">
                  <select
                    value={f.column}
                    onChange={(e) => {
                      const newCol = columns.find((c) => c.name === e.target.value);
                      updateFilter(f.id, {
                        column: e.target.value, type: defaultType(newCol),
                        values: undefined, pattern: undefined, min_val: undefined,
                        max_val: undefined, start: undefined, end: undefined,
                      });
                    }}
                    className="mt-0.5 rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary"
                  >
                    {columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>

                  <select
                    value={f.type}
                    onChange={(e) => updateFilter(f.id, {
                      type: e.target.value,
                      values: undefined, pattern: undefined, min_val: undefined,
                      max_val: undefined, start: undefined, end: undefined,
                    })}
                    title="Filter type"
                    className="mt-0.5 rounded border border-border bg-deep px-2 py-1 text-xs text-text-secondary"
                  >
                    {TYPE_OPTIONS.map((t) => (
                      <option key={t.id} value={t.id}>{t.label}</option>
                    ))}
                  </select>

                  {f.type === 'text' && (
                    <input
                      type="text"
                      placeholder="Search pattern..."
                      value={f.pattern ?? ''}
                      onChange={(e) => updateFilter(f.id, { pattern: e.target.value })}
                      className="mt-0.5 flex-1 rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary placeholder:text-text-muted"
                    />
                  )}

                  {f.type === 'numeric' && (
                    <div className="mt-0.5 flex items-center gap-1">
                      <input
                        type="number"
                        placeholder="Min"
                        value={f.min_val ?? ''}
                        onChange={(e) => updateFilter(f.id, { min_val: e.target.value ? Number(e.target.value) : undefined })}
                        className="w-20 rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary"
                      />
                      <span className="text-text-muted">-</span>
                      <input
                        type="number"
                        placeholder="Max"
                        value={f.max_val ?? ''}
                        onChange={(e) => updateFilter(f.id, { max_val: e.target.value ? Number(e.target.value) : undefined })}
                        className="w-20 rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary"
                      />
                    </div>
                  )}

                  {f.type === 'date' && (
                    <div className="mt-0.5 flex items-center gap-1">
                      <input
                        type="date"
                        value={f.start ?? ''}
                        onChange={(e) => updateFilter(f.id, { start: e.target.value || undefined })}
                        className="rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary"
                      />
                      <span className="text-text-muted">→</span>
                      <input
                        type="date"
                        value={f.end ?? ''}
                        onChange={(e) => updateFilter(f.id, { end: e.target.value || undefined })}
                        className="rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary"
                      />
                    </div>
                  )}

                  {f.type === 'categorical' && (
                    <CategoricalValuePicker
                      column={f.column}
                      selected={f.values ?? []}
                      onChange={(values) => updateFilter(f.id, { values })}
                      fallbackValues={colInfo?.top_values ?? []}
                    />
                  )}

                  <button
                    onClick={() => removeFilter(f.id)}
                    className="mt-1.5 text-text-muted hover:text-danger"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <label className="mt-1.5 flex items-center gap-1.5 text-[10px] text-text-muted">
                  <input
                    type="checkbox"
                    checked={!!f.drop_null}
                    onChange={(e) => updateFilter(f.id, { drop_null: e.target.checked })}
                    className="accent-accent"
                  />
                  Drop null / blank rows for this column
                </label>
                </div>
              );
            })}
          </div>

          <div className="mt-3 flex gap-2">
            <button
              onClick={addFilter}
              className="flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs text-text-secondary hover:bg-elevated"
            >
              <Plus className="h-3 w-3" /> Add Filter
            </button>
            <button
              onClick={handleApply}
              className="rounded bg-accent-dim px-3 py-1.5 text-xs font-medium text-white hover:bg-accent"
            >
              Apply Filters
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
