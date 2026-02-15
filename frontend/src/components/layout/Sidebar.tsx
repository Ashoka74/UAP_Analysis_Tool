import {
  LayoutDashboard,
  Database,
  BrainCircuit,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  Hexagon,
} from 'lucide-react';
import { useStore } from '../../store/useStore';
import type { PageId } from '../../types';

const navItems: { id: PageId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
  { id: 'data', label: 'Data Explorer', icon: Database },
  { id: 'analysis', label: 'Analysis', icon: BrainCircuit },
  { id: 'query', label: 'AI Query', icon: MessageSquare },
];

export function Sidebar() {
  const { currentPage, setPage, sidebarCollapsed, toggleSidebar } = useStore();

  return (
    <aside
      className={`flex h-full flex-col border-r border-border bg-abyss transition-all duration-200 ${
        sidebarCollapsed ? 'w-16' : 'w-56'
      }`}
    >
      {/* Logo */}
      <div className="flex h-14 items-center gap-2.5 border-b border-border px-4">
        <Hexagon className="h-6 w-6 shrink-0 text-accent" />
        {!sidebarCollapsed && (
          <span className="text-sm font-bold tracking-wide text-text-primary">
            UAP FOUNDRY
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-1 p-2">
        {navItems.map(({ id, label, icon: Icon }) => {
          const active = currentPage === id;
          return (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors ${
                active
                  ? 'bg-accent-dim/30 text-accent-bright'
                  : 'text-text-secondary hover:bg-elevated hover:text-text-primary'
              }`}
            >
              <Icon className="h-4.5 w-4.5 shrink-0" />
              {!sidebarCollapsed && <span>{label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Collapse */}
      <div className="border-t border-border p-2">
        <button
          onClick={toggleSidebar}
          className="flex w-full items-center justify-center rounded-md py-2 text-text-muted hover:bg-elevated hover:text-text-secondary"
        >
          {sidebarCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>
    </aside>
  );
}
