import { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react';
import type { DataResponse } from '../../types';

interface DataTableProps {
  data: DataResponse;
  maxHeight?: string;
}

type SortDir = 'asc' | 'desc' | null;

export function DataTable({ data, maxHeight = '500px' }: DataTableProps) {
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>(null);
  const [page, setPage] = useState(0);
  const pageSize = 100;

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir((d) => (d === 'asc' ? 'desc' : d === 'desc' ? null : 'asc'));
      if (sortDir === 'desc') setSortCol(null);
    } else {
      setSortCol(col);
      setSortDir('asc');
    }
    setPage(0);
  };

  const sorted = useMemo(() => {
    if (!sortCol || !sortDir) return data.rows;
    return [...data.rows].sort((a, b) => {
      const va = a[sortCol];
      const vb = b[sortCol];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'number' && typeof vb === 'number') {
        return sortDir === 'asc' ? va - vb : vb - va;
      }
      const sa = String(va);
      const sb = String(vb);
      return sortDir === 'asc' ? sa.localeCompare(sb) : sb.localeCompare(sa);
    });
  }, [data.rows, sortCol, sortDir]);

  const paged = sorted.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(sorted.length / pageSize);

  return (
    <div>
      <div className="overflow-auto" style={{ maxHeight }}>
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-deep">
            <tr>
              <th className="border-b border-border px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                #
              </th>
              {data.columns.map((col) => (
                <th
                  key={col}
                  onClick={() => handleSort(col)}
                  className="cursor-pointer select-none border-b border-border px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-text-muted hover:text-text-secondary"
                >
                  <div className="flex items-center gap-1">
                    <span className="truncate">{col}</span>
                    {sortCol === col ? (
                      sortDir === 'asc' ? (
                        <ChevronUp className="h-3 w-3 text-accent" />
                      ) : (
                        <ChevronDown className="h-3 w-3 text-accent" />
                      )
                    ) : (
                      <ChevronsUpDown className="h-3 w-3 opacity-30" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((row, idx) => (
              <tr
                key={idx}
                className="border-b border-border/30 transition-colors hover:bg-elevated/50"
              >
                <td className="px-3 py-1.5 text-text-muted">{page * pageSize + idx + 1}</td>
                {data.columns.map((col) => (
                  <td key={col} className="max-w-48 truncate px-3 py-1.5 text-text-secondary">
                    {row[col] != null ? String(row[col]) : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between border-t border-border px-4 py-2">
        <span className="text-xs text-text-muted">
          Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, sorted.length)} of{' '}
          {sorted.length} rows ({data.total_rows} total)
        </span>
        <div className="flex items-center gap-1">
          <button
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
            className="rounded border border-border px-2 py-1 text-xs text-text-secondary hover:bg-elevated disabled:opacity-30"
          >
            Prev
          </button>
          <span className="px-2 text-xs text-text-muted">
            {page + 1} / {totalPages}
          </span>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
            className="rounded border border-border px-2 py-1 text-xs text-text-secondary hover:bg-elevated disabled:opacity-30"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
