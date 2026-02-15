import { useState } from 'react';
import { Filter, X, Plus } from 'lucide-react';
import type { ColumnStat } from '../../types';

interface ActiveFilter {
  id: number;
  column: string;
  type: string;
  values?: string[];
  min_val?: number;
  max_val?: number;
  pattern?: string;
}

interface FilterPanelProps {
  columns: ColumnStat[];
  onApply: (filters: ActiveFilter[]) => void;
}

let filterId = 0;

export function FilterPanel({ columns, onApply }: FilterPanelProps) {
  const [filters, setFilters] = useState<ActiveFilter[]>([]);
  const [open, setOpen] = useState(false);

  const addFilter = () => {
    if (columns.length === 0) return;
    const col = columns[0];
    const type = col.top_values ? 'categorical' : col.min != null ? 'numeric' : 'text';
    setFilters([...filters, { id: ++filterId, column: col.name, type }]);
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
                <div key={f.id} className="flex items-center gap-2 rounded bg-raised p-2">
                  <select
                    value={f.column}
                    onChange={(e) => {
                      const newCol = columns.find((c) => c.name === e.target.value);
                      const type = newCol?.top_values ? 'categorical' : newCol?.min != null ? 'numeric' : 'text';
                      updateFilter(f.id, { column: e.target.value, type, values: undefined, pattern: undefined, min_val: undefined, max_val: undefined });
                    }}
                    className="rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary"
                  >
                    {columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>

                  {f.type === 'text' && (
                    <input
                      type="text"
                      placeholder="Search pattern..."
                      value={f.pattern ?? ''}
                      onChange={(e) => updateFilter(f.id, { pattern: e.target.value })}
                      className="flex-1 rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary placeholder:text-text-muted"
                    />
                  )}

                  {f.type === 'numeric' && (
                    <div className="flex items-center gap-1">
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

                  {f.type === 'categorical' && colInfo?.top_values && (
                    <select
                      multiple
                      value={f.values ?? []}
                      onChange={(e) => {
                        const selected = Array.from(e.target.selectedOptions, (o) => o.value);
                        updateFilter(f.id, { values: selected });
                      }}
                      className="max-h-20 flex-1 rounded border border-border bg-deep px-2 py-1 text-xs text-text-primary"
                    >
                      {colInfo.top_values.map((tv) => (
                        <option key={tv.value} value={tv.value}>
                          {tv.value} ({tv.count})
                        </option>
                      ))}
                    </select>
                  )}

                  <button
                    onClick={() => removeFilter(f.id)}
                    className="text-text-muted hover:text-danger"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
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
