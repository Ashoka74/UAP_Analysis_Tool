import type { ReactNode } from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  icon: ReactNode;
  trend?: string;
  color?: string;
}

export function StatCard({ label, value, icon, trend, color = 'text-accent' }: StatCardProps) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-text-muted">{label}</p>
          <p className={`mt-1 text-2xl font-bold ${color}`}>{value}</p>
          {trend && <p className="mt-1 text-xs text-text-secondary">{trend}</p>}
        </div>
        <div className="rounded-md bg-elevated p-2 text-text-muted">{icon}</div>
      </div>
    </div>
  );
}
