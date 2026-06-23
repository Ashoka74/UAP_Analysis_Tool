interface StatusBadgeProps {
  status: 'success' | 'warning' | 'danger' | 'info' | 'idle';
  label: string;
  pulse?: boolean;
}

const colorMap = {
  success: 'bg-success/20 text-success border-success/30',
  warning: 'bg-warning/20 text-warning border-warning/30',
  danger: 'bg-danger/20 text-danger border-danger/30',
  info: 'bg-accent/20 text-accent border-accent/30',
  idle: 'bg-border/50 text-text-muted border-border',
};

export function StatusBadge({ status, label, pulse }: StatusBadgeProps) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${colorMap[status]}`}>
      <span className={`h-1.5 w-1.5 rounded-full bg-current ${pulse ? 'animate-pulse' : ''}`} />
      {label}
    </span>
  );
}
