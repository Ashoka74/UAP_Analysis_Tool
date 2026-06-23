import { useState, useMemo, useEffect } from 'react';
import {
  ChevronUp,
  ChevronDown,
  ChevronsUpDown,
  Maximize2,
  Minimize2,
  Download,
  Eye,
  Search,
  X,
  FileText
} from 'lucide-react';
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
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRow, setSelectedRow] = useState<any | null>(null);
  const pageSize = 100;

  // Handle ESC key for full screen
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullScreen) setIsFullScreen(false);
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isFullScreen]);

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

  const filtered = useMemo(() => {
    if (!searchTerm) return data.rows;
    const lower = searchTerm.toLowerCase();
    return data.rows.filter(row =>
      Object.values(row).some(val => String(val).toLowerCase().includes(lower))
    );
  }, [data.rows, searchTerm]);

  const sorted = useMemo(() => {
    if (!sortCol || !sortDir) return filtered;
    return [...filtered].sort((a, b) => {
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
  }, [filtered, sortCol, sortDir]);

  const paged = sorted.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(sorted.length / pageSize);

  const exportToCSV = () => {
    const headers = data.columns.join(',');
    const rows = sorted.map(row =>
      data.columns.map(col => {
        const val = row[col] == null ? '' : String(row[col]);
        return `"${val.replace(/"/g, '""')}"`;
      }).join(',')
    );
    const csvContent = [headers, ...rows].join('\n');
    // Use a Blob URL, not a data: URI — browsers cap data: URIs at a few MB,
    // which silently blocks large exports. A leading BOM keeps Excel in UTF-8.
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `uap_data_export_${new Date().getTime()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const TableContent = (
    <div className={`flex flex-col bg-surface ${isFullScreen ? 'h-full w-full fixed inset-0 z-[100] p-6 animate-in fade-in zoom-in duration-200' : 'relative'}`}>

      {/* Table Header / Controls */}
      <div className="flex items-center justify-between gap-4 mb-3">
        <div className="flex items-center gap-2 flex-1 max-w-md relative">
          <Search className="h-4 w-4 absolute left-3 text-text-muted" />
          <input
            type="text"
            placeholder="Search current data..."
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); setPage(0); }}
            className="w-full bg-elevated border border-border rounded-md pl-9 pr-4 py-1.5 text-xs focus:ring-1 focus:ring-accent outline-none"
          />
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={exportToCSV}
            title="Export to CSV"
            className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-elevated border border-border text-xs text-text-secondary hover:bg-accent hover:text-white transition-all shadow-sm"
          >
            <Download className="h-3.5 w-3.5" />
            Export
          </button>
          <button
            onClick={() => setIsFullScreen(!isFullScreen)}
            title={isFullScreen ? "Exit Full Screen" : "Full Screen"}
            className="p-1.5 rounded-md bg-elevated border border-border text-text-secondary hover:bg-purple hover:text-white transition-all shadow-sm"
          >
            {isFullScreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
          {isFullScreen && (
            <button
              onClick={() => setIsFullScreen(false)}
              className="p-1.5 rounded-md bg-danger/10 text-danger hover:bg-danger hover:text-white transition-all"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto border border-border rounded-lg bg-deep/50 shadow-inner" style={{ maxHeight: isFullScreen ? 'calc(100vh - 180px)' : maxHeight }}>
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-deep border-b border-border shadow-sm">
            <tr>
              <th className="px-3 py-2.5 text-left text-[10px] font-bold uppercase tracking-wider text-text-muted bg-deep">
                #
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-bold uppercase tracking-wider text-text-muted bg-deep">
                Actions
              </th>
              {data.columns.map((col) => (
                <th
                  key={col}
                  onClick={() => handleSort(col)}
                  className="cursor-pointer select-none px-3 py-2.5 text-left text-[10px] font-bold uppercase tracking-wider text-text-muted hover:text-accent transition-colors bg-deep"
                >
                  <div className="flex items-center gap-1.5">
                    <span className="truncate">{col}</span>
                    {sortCol === col ? (
                      sortDir === 'asc' ? (
                        <ChevronUp className="h-3 w-3 text-accent" />
                      ) : (
                        <ChevronDown className="h-3 w-3 text-accent" />
                      )
                    ) : (
                      <ChevronsUpDown className="h-3 w-3 opacity-20" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border/20">
            {paged.map((row, idx) => (
              <tr
                key={idx}
                className="group border-b border-border/30 transition-colors hover:bg-accent/5"
              >
                <td className="px-3 py-2 text-text-muted/60 font-mono text-[10px]">{page * pageSize + idx + 1}</td>
                <td className="px-3 py-2 text-center">
                  <button
                    onClick={() => setSelectedRow(row)}
                    className="p-1 rounded bg-elevated border border-border text-text-muted hover:text-purple hover:border-purple/50 transition-all opacity-0 group-hover:opacity-100"
                    title="View Details"
                  >
                    <Eye className="h-3.5 w-3.5" />
                  </button>
                </td>
                {data.columns.map((col) => {
                  const val = row[col] != null ? String(row[col]) : '';
                  return (
                    <td
                      key={col}
                      className="max-w-48 truncate px-3 py-2 text-text-secondary font-medium tracking-tight"
                      title={val.length > 30 ? val : ''}
                    >
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        {paged.length === 0 && (
          <div className="py-20 text-center flex flex-col items-center gap-3">
            <div className="p-4 rounded-full bg-elevated border border-border">
              <Search className="h-8 w-8 text-text-muted" />
            </div>
            <div>
              <p className="text-text-primary font-medium">No results found</p>
              <p className="text-xs text-text-muted">Try adjusting your search terms</p>
            </div>
          </div>
        )}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between border-t border-border mt-3 pt-3">
        <div className="flex items-center gap-4">
          <span className="text-[11px] text-text-muted font-medium bg-elevated px-2 py-1 rounded border border-border/50">
            PAGE {page + 1} OF {totalPages || 1}
          </span>
          <span className="text-[11px] text-text-muted">
            Showing <span className="text-text-primary">{page * pageSize + 1}–{Math.min((page + 1) * pageSize, sorted.length)}</span> of{' '}
            <span className="text-text-primary font-bold">{sorted.length}</span> rows
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
            className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-xs font-semibold text-text-secondary bg-surface hover:bg-elevated transition-colors disabled:opacity-30 disabled:hover:bg-surface"
          >
            Previous
          </button>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
            className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-xs font-semibold text-text-secondary bg-surface hover:bg-elevated transition-colors disabled:opacity-30 disabled:hover:bg-surface"
          >
            Next
          </button>
        </div>
      </div>

      {/* Row Detail Modal */}
      {selectedRow && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="w-full max-w-2xl bg-surface border border-border rounded-xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-deep/50">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple/20 text-purple">
                  <FileText className="h-5 w-5" />
                </div>
                <h3 className="font-bold text-text-primary">Row Details</h3>
              </div>
              <button
                onClick={() => setSelectedRow(null)}
                className="p-1.5 rounded-full hover:bg-elevated text-text-muted hover:text-text-primary transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="max-h-[70vh] overflow-auto p-6">
              <div className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2">
                {data.columns.map(col => (
                  <div key={col} className="space-y-1 group">
                    <label className="text-[10px] font-bold uppercase tracking-widest text-text-muted group-hover:text-accent transition-colors">
                      {col}
                    </label>
                    <div className="p-3 bg-elevated rounded-lg border border-border/50 text-sm text-text-secondary break-words font-medium">
                      {selectedRow[col] != null ? String(selectedRow[col]) : <span className="italic opacity-30">null</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="px-6 py-4 bg-deep/30 border-t border-border flex justify-end">
              <button
                onClick={() => setSelectedRow(null)}
                className="px-6 py-2 rounded-lg bg-accent text-white font-bold text-sm hover:scale-[1.02] transform transition-all shadow-lg active:scale-95"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  return TableContent;
}
